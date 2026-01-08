"""
Level Graph Navigator

Navigates level.ai graph for pathfinding and vertex position lookups.
"""

import struct
import math
import numpy as np
import mmap
from collections import deque
from typing import List, Tuple, Optional

from utils import logError
from constants import FLOAT_EPSILON


class LevelGraphNavigator:
    """Navigate level.ai graph for pathfinding"""

    HEADER_SIZE = 56  # Header includes: version, count, size, size_y, aabb (24), guid (16)
    NODE_SIZE = 23

    def __init__(self, level_ai_path: str):
        self.level_ai_path = level_ai_path
        self.file_handle = None
        self.mmap_handle = None
        self._load_header()
        self._build_adjacency()
        self._build_spatial_index()

    def __del__(self):
        if self.mmap_handle:
            self.mmap_handle.close()
        if self.file_handle:
            self.file_handle.close()

    def _load_header(self):
        with open(self.level_ai_path, 'rb') as f:
            header_data = f.read(self.HEADER_SIZE)

        # Header is 56 bytes: version, count, size, size_y, aabb (6 floats), guid (16 bytes)
        (version, count, cell_size, cell_size_y,
         min_x, min_y, min_z, max_x, max_y, max_z,
         guid0, guid1) = struct.unpack('<IIff6fQQ', header_data)

        self.header = {
            'version': version,
            'vertex_count': count,
            'cell_size': cell_size,
            'cell_size_y': cell_size_y,
            'min': (min_x, min_y, min_z),
            'max': (max_x, max_y, max_z),
            'guid': bytes(struct.pack('<QQ', guid0, guid1))
        }

    def _build_adjacency(self):
        vertex_count = self.header['vertex_count']
        edges = []

        with open(self.level_ai_path, 'rb') as f:
            f.seek(self.HEADER_SIZE)

            for vertex_id in range(vertex_count):
                node_data = f.read(self.NODE_SIZE)
                links = self._parse_links(node_data)

                for link in links:
                    if self._is_valid_link(link, vertex_count):
                        edges.append((vertex_id, link))

        edges.sort(key=lambda x: x[0])

        self.edge_destinations = np.array([dst for src, dst in edges], dtype=np.uint32)
        self.row_ptr = np.zeros(vertex_count + 1, dtype=np.uint32)

        current_vertex = 0
        for i, (src, dst) in enumerate(edges):
            while current_vertex < src:
                self.row_ptr[current_vertex + 1] = i
                current_vertex += 1
            self.row_ptr[src + 1] = i + 1

        while current_vertex < vertex_count:
            self.row_ptr[current_vertex + 1] = len(edges)
            current_vertex += 1

    def _parse_links(self, node_data: bytes) -> List[int]:
        l0 = struct.unpack_from("<I", node_data, 0)[0] & 0x007FFFFF
        l1 = (struct.unpack_from("<I", node_data, 2)[0] >> 7) & 0x007FFFFF
        l2 = (struct.unpack_from("<I", node_data, 5)[0] >> 6) & 0x007FFFFF
        l3 = (struct.unpack_from("<I", node_data, 8)[0] >> 5) & 0x007FFFFF
        return [l0, l1, l2, l3]

    def _is_valid_link(self, link: int, vertex_count: int) -> bool:
        return link != 0x7FFFFF and link < vertex_count

    def _build_spatial_index(self):
        """Build a spatial index mapping XZ grid cells to vertex IDs for fast lookup"""
        vertex_count = self.header['vertex_count']
        min_x = self.header['min'][0]
        min_z = self.header['min'][2]
        max_z = self.header['max'][2]
        cell_size = self.header['cell_size']

        # Calculate row length (Z dimension) - must match game's formula exactly
        # Game uses: iFloor((max_z - min_z) / cell_size + FLOAT_EPSILON + 1.5f)
        self.row_length = int((max_z - min_z) / cell_size + FLOAT_EPSILON + 1.5)

        # Build dictionary: xz_packed -> list of vertex_ids
        # Multiple vertices can share the same XZ cell (different heights)
        # Also store vertex positions for distance calculations
        self.xz_to_vertices = {}  # Maps xz -> list of vertex_ids
        self.vertex_positions = np.zeros((vertex_count, 3), dtype=np.float32)

        with open(self.level_ai_path, 'rb') as f:
            f.seek(self.HEADER_SIZE)

            for vertex_id in range(vertex_count):
                node_data = f.read(self.NODE_SIZE)

                # Extract xz packed coordinate
                xz = struct.unpack_from("<I", node_data, 18)[0] & 0x00FFFFFF
                y_quant = struct.unpack_from("<H", node_data, 21)[0]

                # Decode position
                x_idx = xz // self.row_length
                z_idx = xz % self.row_length
                x = x_idx * cell_size + min_x
                z = z_idx * cell_size + min_z

                # Y uses cell_size_y (factor_y) not bbox range - match game formula
                y_normalized = y_quant / 65535.0
                factor_y = self.header['cell_size_y']
                y = y_normalized * factor_y + self.header['min'][1]

                # Store in spatial index - append to list for this cell
                if xz not in self.xz_to_vertices:
                    self.xz_to_vertices[xz] = []
                self.xz_to_vertices[xz].append(vertex_id)
                self.vertex_positions[vertex_id] = (x, y, z)

    def get_neighbors(self, vertex_id: int) -> np.ndarray:
        if not self.valid_vertex_id(vertex_id):
            return np.array([], dtype=np.uint32)
        start = self.row_ptr[vertex_id]
        end = self.row_ptr[vertex_id + 1]
        return self.edge_destinations[start:end]

    def bfs_distances(self, start_vertex: int, max_distance: Optional[int] = None) -> np.ndarray:
        vertex_count = self.header['vertex_count']
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
                neighbors = self.get_neighbors(vertex)
                for neighbor in neighbors:
                    if distances[neighbor] == INFINITY:
                        distances[neighbor] = curr_dist + 1
                        next_fringe.append(int(neighbor))

            current_fringe = next_fringe
            curr_dist += 1

        return distances

    def bfs_path_distance(self, start_vertex: int, end_vertex: int) -> Tuple[float, int]:
        """
        Calculate actual path distance between two vertices via BFS.

        Uses BFS to find the shortest path (by hop count) through the level.ai
        graph, then sums the Euclidean distances along each edge of the path.

        Args:
            start_vertex: Starting level vertex ID
            end_vertex: Target level vertex ID

        Returns:
            (path_distance_meters, hop_count) or (inf, -1) if unreachable
        """
        if not self.valid_vertex_id(start_vertex) or not self.valid_vertex_id(end_vertex):
            return (float('inf'), -1)

        if start_vertex == end_vertex:
            return (0.0, 0)

        # BFS with parent tracking
        vertex_count = self.header['vertex_count']
        INFINITY = np.iinfo(np.uint32).max
        parent = np.full(vertex_count, -1, dtype=np.int32)
        distances = np.full(vertex_count, INFINITY, dtype=np.uint32)

        distances[start_vertex] = 0
        parent[start_vertex] = start_vertex
        queue = deque([start_vertex])

        while queue:
            current = queue.popleft()
            if current == end_vertex:
                break
            for neighbor in self.get_neighbors(current):
                if distances[neighbor] == INFINITY:
                    distances[neighbor] = distances[current] + 1
                    parent[neighbor] = current
                    queue.append(int(neighbor))

        if distances[end_vertex] == INFINITY:
            return (float('inf'), -1)

        # Reconstruct path and calculate actual distance
        path = []
        current = end_vertex
        while current != start_vertex:
            path.append(current)
            current = parent[current]
        path.append(start_vertex)
        path.reverse()

        # Sum edge distances
        total_distance = 0.0
        for i in range(len(path) - 1):
            pos1 = self.vertex_positions[path[i]]
            pos2 = self.vertex_positions[path[i + 1]]
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            dz = pos2[2] - pos1[2]
            total_distance += math.sqrt(dx*dx + dy*dy + dz*dz)

        return (total_distance, len(path) - 1)

    def get_vertex_position(self, vertex_id: int) -> Tuple[float, float, float]:
        if not self.valid_vertex_id(vertex_id):
            return (0.0, 0.0, 0.0)

        # Use cached positions if available (from spatial index)
        if hasattr(self, 'vertex_positions'):
            pos = self.vertex_positions[vertex_id]
            return (float(pos[0]), float(pos[1]), float(pos[2]))

        if not self.mmap_handle:
            self.file_handle = open(self.level_ai_path, 'rb')
            self.mmap_handle = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)

        offset = self.HEADER_SIZE + vertex_id * self.NODE_SIZE
        node_data = self.mmap_handle[offset:offset + self.NODE_SIZE]

        xz = struct.unpack_from("<I", node_data, 18)[0] & 0x00FFFFFF
        y_quant = struct.unpack_from("<H", node_data, 21)[0]

        # X-Ray uses linear index encoding:
        # pxz = x_cell * row_length + z_cell
        # where row_length = number of cells in Z direction
        # See level_graph_inline.h vertex_position()
        min_x = self.header['min'][0]
        min_z = self.header['min'][2]
        max_z = self.header['max'][2]
        cell_size = self.header['cell_size']

        # row_length is the Z dimension (number of columns) - match game formula exactly
        row_length = int((max_z - min_z) / cell_size + FLOAT_EPSILON + 1.5)

        # Decode x and z cell indices
        x_idx = xz // row_length
        z_idx = xz % row_length

        # Convert to world coordinates
        x = x_idx * cell_size + min_x
        z = z_idx * cell_size + min_z

        # Y is quantized to 16-bit, uses cell_size_y (factor_y) not bbox range
        y_normalized = y_quant / 65535.0
        factor_y = self.header['cell_size_y']  # This is size_y / factor_y in game
        y_pos = y_normalized * factor_y + self.header['min'][1]

        return (x, y_pos, z)

    def find_nearest_vertex(self, position: Tuple[float, float, float]) -> int:
        """
        Find the nearest level.ai vertex to the given world position.
        Uses the spatial index for fast grid-based lookup with spiral search.

        For positions outside the AI mesh (buildings, water, etc.), this will
        expand the search until it finds the nearest walkable vertex.
        """
        x, y, z = position
        cell_size = self.header['cell_size']
        min_x = self.header['min'][0]
        min_z = self.header['min'][2]
        max_x = self.header['max'][0]
        max_z = self.header['max'][2]

        # Calculate target grid cell
        target_x_idx = int((x - min_x) / cell_size)
        target_z_idx = int((z - min_z) / cell_size)

        # Calculate maximum possible radius (diagonal of entire level)
        max_x_cells = int((max_x - min_x) / cell_size) + 1
        max_z_cells = int((max_z - min_z) / cell_size) + 1
        max_radius = max(max_x_cells, max_z_cells) + 100  # Extra margin for safety

        # Search in expanding squares around target cell
        best_vertex = None
        best_dist_sq = float('inf')

        for radius in range(max_radius + 1):
            # Search all cells at this radius
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

                    # Calculate packed xz coordinate
                    xz = x_idx * self.row_length + z_idx

                    if xz in self.xz_to_vertices:
                        # Check ALL vertices at this XZ cell (there may be multiple at different heights)
                        for vertex_id in self.xz_to_vertices[xz]:
                            vpos = self.vertex_positions[vertex_id]
                            dx_pos = vpos[0] - x
                            dy_pos = vpos[1] - y
                            dz_pos = vpos[2] - z
                            dist_sq = dx_pos * dx_pos + dy_pos * dy_pos + dz_pos * dz_pos

                            if dist_sq < best_dist_sq:
                                best_dist_sq = dist_sq
                                best_vertex = vertex_id

                        found_any_at_radius = True

            # If we found vertices at this radius and best is close enough, stop searching
            if best_vertex is not None and found_any_at_radius:
                # The minimum possible distance at radius+1 is (radius+1)*cell_size
                # If our best distance is less than that, we've found the optimal
                min_possible_dist_sq = ((radius + 1) * cell_size) ** 2
                if best_dist_sq <= min_possible_dist_sq:
                    break

        if best_vertex is None:
            # This should only happen if level.ai has no vertices at all
            logError(f"ERROR: No vertex found in entire level for position ({x:.1f}, {y:.1f}, {z:.1f})")
            return 0

        return best_vertex

    def valid_vertex_id(self, vertex_id: int) -> bool:
        return 0 <= vertex_id < self.header['vertex_count']
