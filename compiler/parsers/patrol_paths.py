"""
Patrol Path Parser

Parses patrol path data from all.spawn chunk 3 or from extracted .patrols files.

Patrol paths define AI waypoint routes for NPCs.

Chunk 3 format (CPatrolPathStorage):
- Chunk 0: patrol_count (u32)
- Chunk 1: Patrol paths
  - Each patrol is a chunk containing:
    - Sub-chunk 0: name (stringZ)
    - Sub-chunk 1: CPatrolPath data (CGraphAbstractSerialize format)

CPatrolPath format (CGraphAbstractSerialize):
- Chunk 0: vertex_count (u32)
- Chunk 1: Vertices (nested chunks, each containing CPatrolPoint)
- Chunk 2: Edges

CPatrolPoint format:
- stringZ name
- Fvector position (12 bytes)
- u32 flags
- u32 level_vertex_id
- u16 game_vertex_id
"""

import struct
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Iterator
from dataclasses import dataclass, field

from .base import read_stringz, ChunkReader


@dataclass
class PatrolPoint:
    """A single point on a patrol path."""
    id: int
    name: str
    position: Tuple[float, float, float]
    flags: int
    level_vertex_id: int
    game_vertex_id: int


@dataclass
class PatrolEdge:
    """An edge between patrol points."""
    target_id: int
    weight: float


@dataclass
class PatrolPath:
    """A complete patrol path with points and edges."""
    name: str
    points: List[PatrolPoint] = field(default_factory=list)
    edges: Dict[int, List[PatrolEdge]] = field(default_factory=dict)

    @property
    def point_count(self) -> int:
        return len(self.points)

    def get_point_by_name(self, name: str) -> Optional[PatrolPoint]:
        """Get a point by its name."""
        for point in self.points:
            if point.name == name:
                return point
        return None


class PatrolPathParser:
    """
    Parser for patrol path data.

    Can parse from:
    - all.spawn chunk 3 (CPatrolPathStorage)
    - Extracted .patrols files

    Usage:
        # From all.spawn
        parser = PatrolPathParser.from_all_spawn(Path("all.spawn"))

        # From extracted patrols file
        parser = PatrolPathParser.from_patrols_file(Path("level.patrols"))

        for patrol in parser.get_patrols():
            print(f"Patrol: {patrol.name} with {patrol.point_count} points")
    """

    PATROL_CHUNK_ID = 3

    def __init__(self):
        """Initialize empty parser."""
        self._patrols: Dict[str, PatrolPath] = {}
        self._raw_data: Dict[str, bytes] = {}  # For preserving binary data

    @classmethod
    def from_all_spawn(cls, filepath: Union[str, Path]) -> 'PatrolPathParser':
        """
        Create parser from all.spawn file.

        Args:
            filepath: Path to all.spawn file

        Returns:
            PatrolPathParser instance
        """
        filepath = Path(filepath)
        parser = cls()

        with open(filepath, 'rb') as f:
            data = f.read()

        # Find patrol chunk (chunk 3)
        reader = ChunkReader(data)
        for chunk in reader:
            if chunk.chunk_id == cls.PATROL_CHUNK_ID:
                parser._parse_patrol_storage(chunk.data)
                break

        return parser

    @classmethod
    def from_patrols_file(cls, filepath: Union[str, Path]) -> 'PatrolPathParser':
        """
        Create parser from extracted .patrols file.

        .patrols file format:
        - u32 patrol_count
        - For each patrol:
          - u16 name_length
          - name (bytes)
          - u32 data_length
          - data (CPatrolPath binary)

        Args:
            filepath: Path to .patrols file

        Returns:
            PatrolPathParser instance
        """
        filepath = Path(filepath)
        parser = cls()

        with open(filepath, 'rb') as f:
            patrol_count = struct.unpack('<I', f.read(4))[0]

            for _ in range(patrol_count):
                name_len = struct.unpack('<H', f.read(2))[0]
                name = f.read(name_len).decode('utf-8')
                data_len = struct.unpack('<I', f.read(4))[0]
                data = f.read(data_len)

                parser._raw_data[name] = data
                patrol = parser._parse_patrol_path(name, data)
                if patrol:
                    parser._patrols[name] = patrol

        return parser

    @classmethod
    def from_chunk_data(cls, data: bytes) -> 'PatrolPathParser':
        """
        Create parser from raw patrol chunk data.

        Args:
            data: Raw chunk 3 data

        Returns:
            PatrolPathParser instance
        """
        parser = cls()
        parser._parse_patrol_storage(data)
        return parser

    def _parse_patrol_storage(self, data: bytes):
        """Parse CPatrolPathStorage format."""
        reader = ChunkReader(data)

        for chunk in reader:
            if chunk.chunk_id == 0:
                # Patrol count
                patrol_count = struct.unpack_from('<I', chunk.data, 0)[0]
            elif chunk.chunk_id == 1:
                # Patrol paths
                self._parse_patrols_chunk(chunk.data)

    def _parse_patrols_chunk(self, data: bytes):
        """Parse the patrol paths chunk."""
        reader = ChunkReader(data)

        for patrol_chunk in reader:
            name, path_data = self._parse_single_patrol_chunk(patrol_chunk.data)
            if name:
                self._raw_data[name] = path_data
                patrol = self._parse_patrol_path(name, path_data)
                if patrol:
                    self._patrols[name] = patrol

    def _parse_single_patrol_chunk(self, data: bytes) -> Tuple[str, bytes]:
        """Parse a single patrol chunk, extracting name and path data."""
        name = ""
        path_data = b""

        reader = ChunkReader(data)
        for chunk in reader:
            if chunk.chunk_id == 0:
                name, _ = read_stringz(chunk.data, 0)
            elif chunk.chunk_id == 1:
                path_data = chunk.data

        return name, path_data

    def _parse_patrol_path(self, name: str, data: bytes) -> Optional[PatrolPath]:
        """Parse CPatrolPath (CGraphAbstractSerialize format)."""
        patrol = PatrolPath(name=name)

        reader = ChunkReader(data)
        for chunk in reader:
            if chunk.chunk_id == 0:
                # Vertex count (we don't really need this since we parse all vertices)
                pass
            elif chunk.chunk_id == 1:
                # Vertices
                self._parse_vertices_chunk(chunk.data, patrol)
            elif chunk.chunk_id == 2:
                # Edges
                self._parse_edges_chunk(chunk.data, patrol)

        return patrol

    def _parse_vertices_chunk(self, data: bytes, patrol: PatrolPath):
        """Parse vertices chunk (nested vertex chunks)."""
        reader = ChunkReader(data)

        for vertex_chunk in reader:
            point = self._parse_vertex_chunk(vertex_chunk.data)
            if point:
                patrol.points.append(point)

    def _parse_vertex_chunk(self, data: bytes) -> Optional[PatrolPoint]:
        """Parse a single vertex chunk with its sub-chunks."""
        point_id = 0
        point_data: Optional[PatrolPoint] = None

        reader = ChunkReader(data)
        for chunk in reader:
            if chunk.chunk_id == 0:
                # Vertex ID
                point_id = struct.unpack_from('<I', chunk.data, 0)[0]
            elif chunk.chunk_id == 1:
                # CPatrolPoint data
                point_data = self._parse_patrol_point(chunk.data)
                if point_data:
                    point_data.id = point_id

        return point_data

    def _parse_patrol_point(self, data: bytes) -> Optional[PatrolPoint]:
        """Parse CPatrolPoint binary data."""
        try:
            offset = 0

            name, offset = read_stringz(data, offset)

            if offset + 12 > len(data):
                return None
            position = struct.unpack_from('<3f', data, offset)
            offset += 12

            if offset + 4 > len(data):
                return None
            flags = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            if offset + 4 > len(data):
                return None
            level_vertex_id = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            if offset + 2 > len(data):
                return None
            game_vertex_id = struct.unpack_from('<H', data, offset)[0]

            return PatrolPoint(
                id=0,  # Will be set by caller
                name=name,
                position=position,
                flags=flags,
                level_vertex_id=level_vertex_id,
                game_vertex_id=game_vertex_id
            )

        except Exception:
            return None

    def _parse_edges_chunk(self, data: bytes, patrol: PatrolPath):
        """Parse edges chunk."""
        offset = 0

        while offset + 4 <= len(data):
            vertex_id = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            if offset + 4 > len(data):
                break

            edge_count = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            edges = []
            for _ in range(edge_count):
                if offset + 8 > len(data):
                    break

                target_id = struct.unpack_from('<I', data, offset)[0]
                weight = struct.unpack_from('<f', data, offset + 4)[0]
                offset += 8

                edges.append(PatrolEdge(target_id=target_id, weight=weight))

            patrol.edges[vertex_id] = edges

    def get_patrols(self) -> Iterator[PatrolPath]:
        """
        Iterate over all patrol paths.

        Yields:
            PatrolPath objects
        """
        return iter(self._patrols.values())

    def get_patrol(self, name: str) -> Optional[PatrolPath]:
        """
        Get a patrol path by name.

        Args:
            name: Patrol name

        Returns:
            PatrolPath if found, None otherwise
        """
        return self._patrols.get(name)

    def get_patrol_names(self) -> List[str]:
        """
        Get list of all patrol names.

        Returns:
            List of patrol names
        """
        return list(self._patrols.keys())

    def get_raw_data(self, name: str) -> Optional[bytes]:
        """
        Get raw binary data for a patrol path.

        Useful for preserving exact binary format when writing back.

        Args:
            name: Patrol name

        Returns:
            Raw binary data if found, None otherwise
        """
        return self._raw_data.get(name)

    @property
    def patrol_count(self) -> int:
        """Number of patrol paths."""
        return len(self._patrols)

    def get_patrols_by_game_vertex(self, game_vertex_id: int) -> List[PatrolPath]:
        """
        Find patrol paths that have points at a specific game vertex.

        Args:
            game_vertex_id: Game vertex ID to search for

        Returns:
            List of PatrolPath objects
        """
        result = []
        for patrol in self._patrols.values():
            for point in patrol.points:
                if point.game_vertex_id == game_vertex_id:
                    result.append(patrol)
                    break
        return result


def read_extracted_patrols(patrols_path: Path) -> Dict[str, bytes]:
    """
    Legacy function: Read patrols from extracted .patrols file.

    This is a compatibility wrapper for code that hasn't migrated to PatrolPathParser.

    Args:
        patrols_path: Path to .patrols file

    Returns:
        Dict mapping patrol names to raw binary data
    """
    parser = PatrolPathParser.from_patrols_file(patrols_path)
    return {name: parser.get_raw_data(name) for name in parser.get_patrol_names()
            if parser.get_raw_data(name) is not None}
