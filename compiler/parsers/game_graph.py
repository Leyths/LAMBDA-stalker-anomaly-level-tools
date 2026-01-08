"""
Game Graph Parser

Parses game graph data from all.spawn chunk 4.
The game graph defines global connectivity between levels and
provides vertices for entity placement.

File format (from game_graph_space.h):
- Header:
  - u8 version
  - u16 vertex_count
  - u32 edge_count
  - u32 death_point_count
  - xrGUID (16 bytes)
  - u8 level_count
- Level definitions (per level):
  - stringZ name
  - Fvector offset (12 bytes)
  - u8 level_id
  - stringZ section
  - xrGUID (16 bytes)
- Vertices (42 bytes each):
  - Fvector local_point (12 bytes)
  - Fvector global_point (12 bytes)
  - u32 packed (level_id:8, level_vertex_id:24)
  - u8[4] vertex_types
  - u32 edge_offset
  - u32 death_point_offset
  - u8 neighbour_count
  - u8 death_point_count
- Edges (6 bytes each):
  - u16 vertex_id
  - f32 distance
- Death points (20 bytes each):
  - Fvector position (12 bytes)
  - u32 level_vertex_id
  - f32 distance
"""

import struct
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Iterator
from dataclasses import dataclass

from .base import read_stringz, ChunkReader


@dataclass
class GameGraphLevel:
    """Level information from game graph header."""
    level_id: int
    name: str
    section: str
    offset: Tuple[float, float, float]
    guid: bytes


@dataclass
class GameGraphVertex:
    """Game graph vertex data."""
    vertex_id: int
    local_point: Tuple[float, float, float]
    global_point: Tuple[float, float, float]
    level_id: int
    level_vertex_id: int
    vertex_types: bytes  # 4 bytes
    edge_offset: int
    death_point_offset: int
    neighbour_count: int
    death_point_count: int


@dataclass
class GameGraphEdge:
    """Game graph edge data."""
    target_vertex_id: int
    distance: float


@dataclass
class GameGraphHeader:
    """Game graph header data."""
    version: int
    vertex_count: int
    edge_count: int
    death_point_count: int
    guid: bytes
    level_count: int


class GameGraphParser:
    """
    Parser for game graph data (all.spawn chunk 4).

    Provides access to:
    - Level definitions
    - Game vertices (compiled graph points)
    - Edges between vertices
    - Death/spawn points

    Usage:
        parser = GameGraphParser.from_all_spawn(Path("all.spawn"))
        for level in parser.get_levels().values():
            print(f"Level {level.level_id}: {level.name}")

        vertex = parser.get_vertex(0)
        edges = parser.get_edges_for_vertex(0)
    """

    VERTEX_SIZE = 42
    EDGE_SIZE = 6
    DEATH_POINT_SIZE = 20
    GAME_GRAPH_CHUNK_ID = 4

    def __init__(self, data: bytes):
        """
        Initialize parser with game graph binary data.

        Args:
            data: Raw game graph chunk data
        """
        self._data = data
        self._header: Optional[GameGraphHeader] = None
        self._levels: Dict[int, GameGraphLevel] = {}
        self._vertices_start: int = 0
        self._vertex_to_level: Dict[int, int] = {}
        self._parse_header()

    @classmethod
    def from_all_spawn(cls, filepath: Union[str, Path]) -> 'GameGraphParser':
        """
        Create parser from all.spawn file.

        Args:
            filepath: Path to all.spawn file

        Returns:
            GameGraphParser instance
        """
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            data = f.read()

        # Find game graph chunk (chunk 4)
        reader = ChunkReader(data)
        for chunk in reader:
            if chunk.chunk_id == cls.GAME_GRAPH_CHUNK_ID:
                return cls(chunk.data)

        raise ValueError(f"Game graph chunk ({cls.GAME_GRAPH_CHUNK_ID}) not found in {filepath}")

    @classmethod
    def from_chunk_data(cls, data: bytes) -> 'GameGraphParser':
        """
        Create parser from raw chunk data.

        Args:
            data: Raw game graph chunk data

        Returns:
            GameGraphParser instance
        """
        return cls(data)

    def _parse_header(self):
        """Parse game graph header and level definitions."""
        offset = 0

        # Header
        version = struct.unpack_from('<B', self._data, offset)[0]
        offset += 1

        vertex_count = struct.unpack_from('<H', self._data, offset)[0]
        offset += 2

        edge_count = struct.unpack_from('<I', self._data, offset)[0]
        offset += 4

        death_point_count = struct.unpack_from('<I', self._data, offset)[0]
        offset += 4

        guid = self._data[offset:offset + 16]
        offset += 16

        level_count = struct.unpack_from('<B', self._data, offset)[0]
        offset += 1

        self._header = GameGraphHeader(
            version=version,
            vertex_count=vertex_count,
            edge_count=edge_count,
            death_point_count=death_point_count,
            guid=guid,
            level_count=level_count
        )

        # Parse level definitions
        for _ in range(level_count):
            level_name, offset = read_stringz(self._data, offset)
            level_offset = struct.unpack_from('<3f', self._data, offset)
            offset += 12
            level_id = struct.unpack_from('<B', self._data, offset)[0]
            offset += 1
            level_section, offset = read_stringz(self._data, offset)
            level_guid = self._data[offset:offset + 16]
            offset += 16

            self._levels[level_id] = GameGraphLevel(
                level_id=level_id,
                name=level_name,
                section=level_section,
                offset=level_offset,
                guid=level_guid
            )

        self._vertices_start = offset

        # Build vertex to level mapping
        for vertex_id in range(vertex_count):
            v_offset = self._vertices_start + vertex_id * self.VERTEX_SIZE
            packed = struct.unpack_from('<I', self._data, v_offset + 24)[0]
            level_id = packed & 0xFF
            self._vertex_to_level[vertex_id] = level_id

    @property
    def header(self) -> GameGraphHeader:
        """Get game graph header."""
        if self._header is None:
            raise RuntimeError("Game graph not loaded")
        return self._header

    @property
    def vertex_count(self) -> int:
        """Number of vertices in the game graph."""
        return self.header.vertex_count

    @property
    def edge_count(self) -> int:
        """Number of edges in the game graph."""
        return self.header.edge_count

    @property
    def level_count(self) -> int:
        """Number of levels in the game graph."""
        return self.header.level_count

    def get_levels(self) -> Dict[int, GameGraphLevel]:
        """
        Get all level definitions.

        Returns:
            Dict mapping level_id to GameGraphLevel
        """
        return self._levels.copy()

    def get_level(self, level_id: int) -> Optional[GameGraphLevel]:
        """
        Get a specific level by ID.

        Args:
            level_id: Level ID

        Returns:
            GameGraphLevel if found, None otherwise
        """
        return self._levels.get(level_id)

    def get_level_by_name(self, name: str) -> Optional[GameGraphLevel]:
        """
        Get a level by name.

        Args:
            name: Level name

        Returns:
            GameGraphLevel if found, None otherwise
        """
        for level in self._levels.values():
            if level.name == name:
                return level
        return None

    def get_vertex(self, vertex_id: int) -> GameGraphVertex:
        """
        Get a specific vertex by ID.

        Args:
            vertex_id: Vertex ID

        Returns:
            GameGraphVertex
        """
        if vertex_id < 0 or vertex_id >= self.header.vertex_count:
            raise IndexError(f"Vertex ID {vertex_id} out of range [0, {self.header.vertex_count})")

        v_offset = self._vertices_start + vertex_id * self.VERTEX_SIZE

        local_point = struct.unpack_from('<3f', self._data, v_offset)
        global_point = struct.unpack_from('<3f', self._data, v_offset + 12)

        packed = struct.unpack_from('<I', self._data, v_offset + 24)[0]
        level_id = packed & 0xFF
        level_vertex_id = (packed >> 8) & 0xFFFFFF

        vertex_types = self._data[v_offset + 28:v_offset + 32]

        edge_offset = struct.unpack_from('<I', self._data, v_offset + 32)[0]
        death_point_offset = struct.unpack_from('<I', self._data, v_offset + 36)[0]
        neighbour_count = struct.unpack_from('<B', self._data, v_offset + 40)[0]
        death_point_count = struct.unpack_from('<B', self._data, v_offset + 41)[0]

        return GameGraphVertex(
            vertex_id=vertex_id,
            local_point=local_point,
            global_point=global_point,
            level_id=level_id,
            level_vertex_id=level_vertex_id,
            vertex_types=vertex_types,
            edge_offset=edge_offset,
            death_point_offset=death_point_offset,
            neighbour_count=neighbour_count,
            death_point_count=death_point_count
        )

    def get_vertices(self) -> Iterator[GameGraphVertex]:
        """
        Iterate over all vertices.

        Yields:
            GameGraphVertex objects
        """
        for vertex_id in range(self.header.vertex_count):
            yield self.get_vertex(vertex_id)

    def get_vertices_for_level(self, level_id: int) -> List[GameGraphVertex]:
        """
        Get all vertices belonging to a specific level.

        Args:
            level_id: Level ID

        Returns:
            List of GameGraphVertex objects
        """
        vertices = []
        for vertex_id, vid_level in self._vertex_to_level.items():
            if vid_level == level_id:
                vertices.append(self.get_vertex(vertex_id))
        return vertices

    def get_edges_for_vertex(self, vertex_id: int) -> List[GameGraphEdge]:
        """
        Get all edges starting from a vertex.

        Args:
            vertex_id: Source vertex ID

        Returns:
            List of GameGraphEdge objects
        """
        vertex = self.get_vertex(vertex_id)

        # Edge offset is relative to the start of vertices
        actual_edge_offset = self._vertices_start + vertex.edge_offset

        edges = []
        for i in range(vertex.neighbour_count):
            e_offset = actual_edge_offset + i * self.EDGE_SIZE
            target_vertex_id = struct.unpack_from('<H', self._data, e_offset)[0]
            distance = struct.unpack_from('<f', self._data, e_offset + 2)[0]
            edges.append(GameGraphEdge(target_vertex_id=target_vertex_id, distance=distance))

        return edges

    def get_level_id_for_vertex(self, vertex_id: int) -> int:
        """
        Get the level ID that a vertex belongs to.

        Args:
            vertex_id: Vertex ID

        Returns:
            Level ID
        """
        return self._vertex_to_level.get(vertex_id, -1)

    def get_vertex_level_ranges(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Get vertex ID ranges for each level.

        Returns:
            Dict mapping level_id to (min_vertex, max_vertex, count)
        """
        ranges: Dict[int, List[int]] = {}

        for vertex_id, level_id in self._vertex_to_level.items():
            if level_id not in ranges:
                ranges[level_id] = [vertex_id, vertex_id, 1]
            else:
                ranges[level_id][0] = min(ranges[level_id][0], vertex_id)
                ranges[level_id][1] = max(ranges[level_id][1], vertex_id)
                ranges[level_id][2] += 1

        return {lid: tuple(vals) for lid, vals in ranges.items()}

    def get_edges_by_level(self) -> Dict[int, List[Dict]]:
        """
        Get all edges grouped by source level.

        Returns:
            Dict mapping level_id to list of edge data dicts
        """
        edges_by_level: Dict[int, List[Dict]] = {}

        for vertex_id in range(self.header.vertex_count):
            vertex = self.get_vertex(vertex_id)
            source_level = vertex.level_id
            source_pos = vertex.local_point

            if source_level not in edges_by_level:
                edges_by_level[source_level] = []

            for edge in self.get_edges_for_vertex(vertex_id):
                target_vertex = self.get_vertex(edge.target_vertex_id)
                target_level = target_vertex.level_id
                target_pos = target_vertex.local_point

                target_level_name = self._levels.get(target_level)
                target_level_name = target_level_name.name if target_level_name else f"level_{target_level}"

                edges_by_level[source_level].append({
                    "source_x": round(source_pos[0], 4),
                    "source_y": round(source_pos[1], 4),
                    "source_z": round(source_pos[2], 4),
                    "target_x": round(target_pos[0], 4),
                    "target_y": round(target_pos[1], 4),
                    "target_z": round(target_pos[2], 4),
                    "distance": round(edge.distance, 4),
                    "target_level": target_level_name
                })

        return edges_by_level

    def get_cross_tables_offset(self) -> int:
        """
        Calculate the offset to where cross-tables start in the data.

        Cross-tables are embedded after the death points in all.spawn chunk 4.
        """
        # Structure: vertices, then edges, then death points, then cross-tables
        vertices_end = self._vertices_start + self.header.vertex_count * self.VERTEX_SIZE
        edges_end = vertices_end + self.header.edge_count * self.EDGE_SIZE
        death_points_end = edges_end + self.header.death_point_count * self.DEATH_POINT_SIZE
        return death_points_end

    def get_cross_table_gvid(self, level_id: int, level_vertex_id: int) -> Optional[int]:
        """
        Get the game vertex ID mapped to a level vertex from embedded cross-table.

        Args:
            level_id: Level ID
            level_vertex_id: Level-local vertex ID

        Returns:
            Game vertex ID if found, None otherwise
        """
        if level_id not in self._levels:
            return None

        cross_tables_start = self.get_cross_tables_offset()
        offset = cross_tables_start

        # Parse cross-tables until we find the one for our level
        # Cross-tables are stored in order by level_id (ascending)
        level_ids = sorted(self._levels.keys())

        for lid in level_ids:
            if offset >= len(self._data):
                return None

            # Read cross-table header
            # u32 total_size (includes this field)
            total_size = struct.unpack_from('<I', self._data, offset)[0]

            if lid == level_id:
                # Found our level's cross-table
                # Header (44 bytes after total_size):
                #   u32 version
                #   u32 level_vertex_count
                #   u32 game_vertex_count
                #   16 bytes level_guid
                #   16 bytes game_guid
                header_offset = offset + 4
                level_vertex_count = struct.unpack_from('<I', self._data, header_offset + 4)[0]

                if level_vertex_id >= level_vertex_count:
                    return None

                # Cells start after header (44 bytes after total_size field)
                cells_offset = offset + 4 + 44

                # Each cell is 6 bytes: u16 game_vertex_id, f32 distance
                cell_offset = cells_offset + level_vertex_id * 6
                game_vertex_id = struct.unpack_from('<H', self._data, cell_offset)[0]

                return game_vertex_id

            # Skip to next cross-table
            offset += total_size

        return None

    def get_cross_table_for_level(self, level_id: int) -> Optional[dict]:
        """
        Parse and return the full cross-table for a level.

        Args:
            level_id: Level ID

        Returns:
            Dict with 'level_vertex_count' and 'gvids' (numpy array of game vertex IDs),
            or None if level not found
        """
        if level_id not in self._levels:
            return None

        cross_tables_start = self.get_cross_tables_offset()
        offset = cross_tables_start

        # Parse cross-tables until we find the one for our level
        level_ids = sorted(self._levels.keys())

        for lid in level_ids:
            if offset >= len(self._data):
                return None

            # Read cross-table header
            total_size = struct.unpack_from('<I', self._data, offset)[0]

            if lid == level_id:
                # Found our level's cross-table
                header_offset = offset + 4
                level_vertex_count = struct.unpack_from('<I', self._data, header_offset + 4)[0]

                # Parse all cells
                cells_offset = offset + 4 + 44
                gvids = []

                for i in range(level_vertex_count):
                    cell_offset = cells_offset + i * 6
                    game_vertex_id = struct.unpack_from('<H', self._data, cell_offset)[0]
                    gvids.append(game_vertex_id)

                return {
                    'level_vertex_count': level_vertex_count,
                    'gvids': gvids
                }

            # Skip to next cross-table
            offset += total_size

        return None
