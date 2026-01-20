"""
Level AI Graph (.ai) Parser

Parses level.ai files containing the navigation mesh for a level.
These files define walkable vertices and their connections for AI pathfinding.

File format:
- Header (56 bytes):
  - u32 version (XRAI_CURRENT_VERSION = 10)
  - u32 vertex_count
  - f32 cell_size (typically 0.7m)
  - f32 cell_size_y
  - Fvector min (3 floats) - bounding box minimum
  - Fvector max (3 floats) - bounding box maximum
  - xrGUID (16 bytes)
- Vertices (23 bytes each):
  - Links to neighbors (packed bits)
  - Position data (quantized)
"""

import struct
import mmap
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from collections import deque


@dataclass
class LevelAIHeader:
    """Level AI graph header data."""
    version: int
    vertex_count: int
    cell_size: float
    cell_size_y: float
    min: Tuple[float, float, float]
    max: Tuple[float, float, float]
    guid: bytes


class LevelAIParser:
    """
    Parser for level.ai navigation mesh files.

    Provides efficient access to level AI graph data including:
    - Vertex positions
    - Neighbor links for pathfinding
    - Spatial indexing for nearest vertex queries
    - BFS distance calculations

    Usage:
        parser = LevelAIParser(Path("level.ai"))
        pos = parser.get_vertex_position(vertex_id=1234)
        neighbors = parser.get_vertex_links(vertex_id=1234)
        nearest = parser.find_nearest_vertex((100.0, 0.0, 200.0))
    """

    XRAI_CURRENT_VERSION = 10
    HEADER_SIZE = 56
    NODE_SIZE = 23

    def __init__(self, filepath: Union[str, Path], build_adjacency: bool = True):
        """
        Load and parse a level.ai file.

        Args:
            filepath: Path to the .ai file
            build_adjacency: If True, build adjacency data for pathfinding (slower load, faster queries)
        """
        self.filepath = Path(filepath)
        self._header: Optional[LevelAIHeader] = None
        self._file_handle = None
        self._mmap_handle = None

        # Adjacency data (built on demand)
        self._edge_destinations: Optional[np.ndarray] = None
        self._row_ptr: Optional[np.ndarray] = None

        # Spatial index (built on demand)
        self._xz_to_vertex: Optional[Dict[int, int]] = None
        self._vertex_positions: Optional[np.ndarray] = None
        self._cover_scores: Optional[np.ndarray] = None
        self._row_length: int = 0

        self._load_header()
        if build_adjacency:
            self._build_adjacency()
            self._build_spatial_index()

    def __del__(self):
        """Clean up file handles."""
        if self._mmap_handle:
            self._mmap_handle.close()
        if self._file_handle:
            self._file_handle.close()

    def _load_header(self):
        """Load header from file."""
        with open(self.filepath, 'rb') as f:
            header_data = f.read(self.HEADER_SIZE)

        if len(header_data) < self.HEADER_SIZE:
            raise ValueError(f"File too small for level.ai header: {len(header_data)} bytes")

        (version, count, cell_size, cell_size_y,
         min_x, min_y, min_z, max_x, max_y, max_z,
         guid0, guid1) = struct.unpack('<IIff6fQQ', header_data)

        self._header = LevelAIHeader(
            version=version,
            vertex_count=count,
            cell_size=cell_size,
            cell_size_y=cell_size_y,
            min=(min_x, min_y, min_z),
            max=(max_x, max_y, max_z),
            guid=bytes(struct.pack('<QQ', guid0, guid1))
        )

    def _build_adjacency(self):
        """Build CSR adjacency structure for efficient neighbor lookup."""
        vertex_count = self._header.vertex_count
        edges = []

        with open(self.filepath, 'rb') as f:
            f.seek(self.HEADER_SIZE)

            for vertex_id in range(vertex_count):
                node_data = f.read(self.NODE_SIZE)
                links = self._parse_links(node_data)

                for link in links:
                    if self._is_valid_link(link, vertex_count):
                        edges.append((vertex_id, link))

        edges.sort(key=lambda x: x[0])

        self._edge_destinations = np.array([dst for src, dst in edges], dtype=np.uint32)
        self._row_ptr = np.zeros(vertex_count + 1, dtype=np.uint32)

        current_vertex = 0
        for i, (src, dst) in enumerate(edges):
            while current_vertex < src:
                self._row_ptr[current_vertex + 1] = i
                current_vertex += 1
            self._row_ptr[src + 1] = i + 1

        while current_vertex < vertex_count:
            self._row_ptr[current_vertex + 1] = len(edges)
            current_vertex += 1

    def _parse_links(self, node_data: bytes) -> List[int]:
        """Parse neighbor links from node data."""
        l0 = struct.unpack_from("<I", node_data, 0)[0] & 0x007FFFFF
        l1 = (struct.unpack_from("<I", node_data, 2)[0] >> 7) & 0x007FFFFF
        l2 = (struct.unpack_from("<I", node_data, 5)[0] >> 6) & 0x007FFFFF
        l3 = (struct.unpack_from("<I", node_data, 8)[0] >> 5) & 0x007FFFFF
        return [l0, l1, l2, l3]

    def _is_valid_link(self, link: int, vertex_count: int) -> bool:
        """Check if a link value is valid."""
        return link != 0x7FFFFF and link < vertex_count

    def _build_spatial_index(self):
        """Build spatial index for fast nearest-vertex lookups."""
        import math
        vertex_count = self._header.vertex_count
        min_x = self._header.min[0]
        min_z = self._header.min[2]
        max_z = self._header.max[2]
        cell_size = self._header.cell_size

        # Row length calculation must match X-Ray engine exactly
        self._row_length = int(math.floor((max_z - min_z) / cell_size + 1.5))
        self._xz_to_vertex = {}
        self._vertex_positions = np.zeros((vertex_count, 3), dtype=np.float32)
        self._cover_scores = np.zeros(vertex_count, dtype=np.float32)

        with open(self.filepath, 'rb') as f:
            f.seek(self.HEADER_SIZE)

            for vertex_id in range(vertex_count):
                node_data = f.read(self.NODE_SIZE)

                xz = struct.unpack_from("<I", node_data, 18)[0] & 0x00FFFFFF
                y_quant = struct.unpack_from("<H", node_data, 21)[0]

                x_idx = xz // self._row_length
                z_idx = xz % self._row_length
                x = x_idx * cell_size + min_x
                z = z_idx * cell_size + min_z

                # Y uses cell_size_y, not bounding box range
                y = self._header.min[1] + y_quant * self._header.cell_size_y / 65535.0

                self._xz_to_vertex[xz] = vertex_id
                self._vertex_positions[vertex_id] = (x, y, z)

                # Parse cover data for visualization (bytes 12-15)
                high_cover = struct.unpack_from("<H", node_data, 12)[0]
                low_cover = struct.unpack_from("<H", node_data, 14)[0]
                # Sum of all 8 nibbles (4 high + 4 low)
                cover_sum = sum((high_cover >> (i * 4)) & 0xF for i in range(4))
                cover_sum += sum((low_cover >> (i * 4)) & 0xF for i in range(4))
                self._cover_scores[vertex_id] = cover_sum

    @property
    def header(self) -> LevelAIHeader:
        """Get level AI header."""
        if self._header is None:
            raise RuntimeError("Level AI not loaded")
        return self._header

    @property
    def vertex_count(self) -> int:
        """Number of vertices in this level AI graph."""
        return self.header.vertex_count

    @property
    def guid(self) -> bytes:
        """Level AI GUID (16 bytes)."""
        return self.header.guid

    def get_vertex_position(self, vertex_id: int) -> Tuple[float, float, float]:
        """
        Get the world position of a vertex.

        Args:
            vertex_id: Vertex ID

        Returns:
            (x, y, z) position tuple
        """
        if not self._valid_vertex_id(vertex_id):
            return (0.0, 0.0, 0.0)

        # Use cached positions if available
        if self._vertex_positions is not None:
            pos = self._vertex_positions[vertex_id]
            return (float(pos[0]), float(pos[1]), float(pos[2]))

        # Fall back to reading from file
        return self._read_vertex_position(vertex_id)

    def _read_vertex_position(self, vertex_id: int) -> Tuple[float, float, float]:
        """Read vertex position directly from file."""
        import math
        if not self._mmap_handle:
            self._file_handle = open(self.filepath, 'rb')
            self._mmap_handle = mmap.mmap(self._file_handle.fileno(), 0, access=mmap.ACCESS_READ)

        offset = self.HEADER_SIZE + vertex_id * self.NODE_SIZE
        node_data = self._mmap_handle[offset:offset + self.NODE_SIZE]

        xz = struct.unpack_from("<I", node_data, 18)[0] & 0x00FFFFFF
        y_quant = struct.unpack_from("<H", node_data, 21)[0]

        min_x = self._header.min[0]
        min_z = self._header.min[2]
        max_z = self._header.max[2]
        cell_size = self._header.cell_size
        cell_size_y = self._header.cell_size_y

        # Row length calculation must match X-Ray engine exactly
        row_length = int(math.floor((max_z - min_z) / cell_size + 1.5))

        x_idx = xz // row_length
        z_idx = xz % row_length
        x = x_idx * cell_size + min_x
        z = z_idx * cell_size + min_z

        # Y uses cell_size_y, not bounding box range
        y = self._header.min[1] + y_quant * cell_size_y / 65535.0

        return (x, y, z)

    def get_vertex_links(self, vertex_id: int) -> List[int]:
        """
        Get neighbor vertex IDs for a vertex.

        Args:
            vertex_id: Vertex ID

        Returns:
            List of neighbor vertex IDs
        """
        if self._edge_destinations is None or self._row_ptr is None:
            raise RuntimeError("Adjacency data not built - initialize with build_adjacency=True")
        if not self._valid_vertex_id(vertex_id):
            return []

        start = self._row_ptr[vertex_id]
        end = self._row_ptr[vertex_id + 1]
        return self._edge_destinations[start:end].tolist()

    def get_all_positions(self) -> np.ndarray:
        """
        Get all vertex positions as a numpy array.

        Returns:
            Nx3 numpy array of (x, y, z) positions
        """
        if self._vertex_positions is None:
            raise RuntimeError("Spatial index not built - initialize with build_adjacency=True")
        return self._vertex_positions.copy()

    def get_all_cover_scores(self) -> np.ndarray:
        """
        Get all vertex cover scores as a numpy array.

        Cover scores are the sum of high and low cover values (0-120 range).
        Used for visualization coloring.

        Returns:
            N-element numpy array of cover scores
        """
        if self._cover_scores is None:
            raise RuntimeError("Spatial index not built - initialize with build_adjacency=True")
        return self._cover_scores.copy()

    def get_vertex_raw_links(self, vertex_id: int) -> List[int]:
        """
        Get raw link values for a vertex (includes invalid link marker 0x7FFFFF).

        Unlike get_vertex_links() which filters to valid neighbors only,
        this returns all 4 link slots as stored in the file.

        Args:
            vertex_id: Vertex ID

        Returns:
            List of 4 link values (0x7FFFFF = no link)
        """
        if not self._valid_vertex_id(vertex_id):
            return [0x7FFFFF, 0x7FFFFF, 0x7FFFFF, 0x7FFFFF]

        if not self._mmap_handle:
            self._file_handle = open(self.filepath, 'rb')
            self._mmap_handle = mmap.mmap(self._file_handle.fileno(), 0, access=mmap.ACCESS_READ)

        offset = self.HEADER_SIZE + vertex_id * self.NODE_SIZE
        node_data = self._mmap_handle[offset:offset + self.NODE_SIZE]
        return self._parse_links(node_data)

    def get_all_links(self) -> np.ndarray:
        """
        Bulk read all vertex links as a (vertex_count, 4) numpy array.

        This is significantly faster than calling get_vertex_raw_links() for each
        vertex individually, as it reads the entire file in a single pass.

        Returns:
            (vertex_count, 4) int32 array where each row contains 4 link values.
            Invalid links are marked with 0x7FFFFF.
        """
        vertex_count = self._header.vertex_count
        links = np.zeros((vertex_count, 4), dtype=np.int32)

        with open(self.filepath, 'rb') as f:
            f.seek(self.HEADER_SIZE)

            for i in range(vertex_count):
                node_data = f.read(self.NODE_SIZE)
                links[i] = self._parse_links(node_data)

        return links

    def find_nearest_vertex(self, position: Tuple[float, float, float]) -> int:
        """
        Find the nearest vertex to a world position.

        Uses the spatial index for fast grid-based lookup with spiral search.
        For positions outside the AI mesh (buildings, water, etc.), this
        expands the search until it finds the nearest walkable vertex.

        Args:
            position: (x, y, z) world position

        Returns:
            Nearest vertex ID
        """
        if self._xz_to_vertex is None or self._vertex_positions is None:
            raise RuntimeError("Spatial index not built - initialize with build_adjacency=True")

        x, y, z = position
        cell_size = self._header.cell_size
        min_x = self._header.min[0]
        min_z = self._header.min[2]
        max_x = self._header.max[0]
        max_z = self._header.max[2]

        # Calculate target grid cell
        target_x_idx = int((x - min_x) / cell_size)
        target_z_idx = int((z - min_z) / cell_size)

        # Calculate maximum search radius
        max_x_cells = int((max_x - min_x) / cell_size) + 1
        max_z_cells = int((max_z - min_z) / cell_size) + 1
        max_radius = max(max_x_cells, max_z_cells) + 100

        best_vertex = None
        best_dist_sq = float('inf')

        for radius in range(max_radius + 1):
            found_any_at_radius = False

            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    # Only check cells on the perimeter of this square
                    if radius > 0 and abs(dx) != radius and abs(dz) != radius:
                        continue

                    x_idx = target_x_idx + dx
                    z_idx = target_z_idx + dz

                    if x_idx < 0 or z_idx < 0:
                        continue

                    xz = x_idx * self._row_length + z_idx

                    if xz in self._xz_to_vertex:
                        vertex_id = self._xz_to_vertex[xz]
                        vpos = self._vertex_positions[vertex_id]
                        dx_pos = vpos[0] - x
                        dy_pos = vpos[1] - y
                        dz_pos = vpos[2] - z
                        dist_sq = dx_pos * dx_pos + dy_pos * dy_pos + dz_pos * dz_pos

                        if dist_sq < best_dist_sq:
                            best_dist_sq = dist_sq
                            best_vertex = vertex_id

                        found_any_at_radius = True

            # If we found vertices at this radius and best is close enough, stop
            if best_vertex is not None and found_any_at_radius:
                min_possible_dist_sq = ((radius + 1) * cell_size) ** 2
                if best_dist_sq <= min_possible_dist_sq:
                    break

        if best_vertex is None:
            return 0

        return best_vertex

    def bfs_distances(self, start_vertex: int, max_distance: Optional[int] = None) -> np.ndarray:
        """
        Calculate BFS distances from a starting vertex to all other vertices.

        Args:
            start_vertex: Starting vertex ID
            max_distance: Optional maximum distance to search (in edge hops)

        Returns:
            Array where distances[i] is the hop distance to vertex i,
            or UINT32_MAX if unreachable
        """
        if self._edge_destinations is None or self._row_ptr is None:
            raise RuntimeError("Adjacency data not built - initialize with build_adjacency=True")

        vertex_count = self._header.vertex_count
        INFINITY = np.iinfo(np.uint32).max
        distances = np.full(vertex_count, INFINITY, dtype=np.uint32)
        distances[start_vertex] = 0

        current_fringe = deque([start_vertex])
        curr_dist = 0

        while current_fringe:
            if max_distance and curr_dist >= max_distance:
                break

            next_fringe = deque()
            for vertex in current_fringe:
                start = self._row_ptr[vertex]
                end = self._row_ptr[vertex + 1]
                neighbors = self._edge_destinations[start:end]

                for neighbor in neighbors:
                    if distances[neighbor] == INFINITY:
                        distances[neighbor] = curr_dist + 1
                        next_fringe.append(int(neighbor))

            current_fringe = next_fringe
            curr_dist += 1

        return distances

    def _valid_vertex_id(self, vertex_id: int) -> bool:
        """Check if a vertex ID is valid."""
        return 0 <= vertex_id < self._header.vertex_count


# Cache for loaded level AI positions
_level_ai_cache: Dict[str, Optional[np.ndarray]] = {}


def load_level_ai_positions(level_ai_path: Path) -> Optional[np.ndarray]:
    """
    Legacy function: Load vertex positions from level.ai with caching.

    This is a compatibility wrapper for code that hasn't migrated to LevelAIParser.

    Args:
        level_ai_path: Path to level.ai file

    Returns:
        Nx3 numpy array of positions, or None on error
    """
    global _level_ai_cache
    cache_key = str(level_ai_path)

    if cache_key in _level_ai_cache:
        return _level_ai_cache[cache_key]

    try:
        parser = LevelAIParser(level_ai_path, build_adjacency=True)
        positions = parser.get_all_positions()
        _level_ai_cache[cache_key] = positions
        return positions
    except Exception:
        _level_ai_cache[cache_key] = None
        return None


def find_nearest_level_vertex(position: Tuple[float, float, float], level_ai_path: Path) -> int:
    """
    Legacy function: Find nearest level vertex to a position.

    This is a compatibility wrapper for code that hasn't migrated to LevelAIParser.

    Args:
        position: (x, y, z) position
        level_ai_path: Path to level.ai file

    Returns:
        Nearest vertex ID, or 0xFFFFFFFF on error
    """
    try:
        positions = load_level_ai_positions(level_ai_path)
        if positions is None:
            return 0xFFFFFFFF

        px, py, pz = position
        dx = positions[:, 0] - np.float32(px)
        dy = positions[:, 1] - np.float32(py)
        dz = positions[:, 2] - np.float32(pz)
        dist_sq = dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2 + dz.astype(np.float64) ** 2
        return int(np.argmin(dist_sq))
    except Exception:
        return 0xFFFFFFFF
