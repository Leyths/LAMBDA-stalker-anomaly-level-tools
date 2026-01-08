"""
Cross Table (.gct) Parser

Parses level.gct cross table files that map level-local vertex IDs
to global game graph vertex IDs.

File format:
- Chunk 0xFFFF (header):
  - u32 version (XRAI_CURRENT_VERSION = 10)
  - u32 level_vertex_count
  - u32 game_vertex_count
  - 16 bytes level_guid
  - 16 bytes game_guid
- Chunk 1 (cells):
  - For each level vertex (level_vertex_count entries):
    - u16 game_vertex_id
    - f32 distance
"""

import struct
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
from dataclasses import dataclass

from .base import ChunkReader


@dataclass
class CrossTableHeader:
    """Cross table header data."""
    version: int
    level_vertex_count: int
    game_vertex_count: int
    level_guid: bytes
    game_guid: bytes


class CrossTableParser:
    """
    Parser for level.gct cross table files.

    Cross tables define the mapping between level-local AI graph vertices
    and global game graph vertices. Each entry contains:
    - Which game vertex this level vertex belongs to
    - The distance from this level vertex to its game vertex

    Usage:
        parser = CrossTableParser(Path("level.gct"))
        game_vertex = parser.get_game_vertex_id(level_vertex_id=1234)
        distance = parser.get_distance(level_vertex_id=1234)
    """

    HEADER_CHUNK_ID = 0xFFFF
    DATA_CHUNK_ID = 1
    CELL_SIZE = 6  # u16 + f32

    def __init__(self, filepath: Union[str, Path]):
        """
        Load and parse a cross table file.

        Args:
            filepath: Path to the .gct file
        """
        self.filepath = Path(filepath)
        self._header: Optional[CrossTableHeader] = None
        self._cells: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        """Load cross table from file."""
        with open(self.filepath, 'rb') as f:
            data = f.read()

        reader = ChunkReader(data)

        for chunk in reader:
            if chunk.chunk_id == self.HEADER_CHUNK_ID:
                self._parse_header(chunk.data)
            elif chunk.chunk_id == self.DATA_CHUNK_ID:
                self._parse_cells(chunk.data)

        if self._header is None:
            raise ValueError(f"No header chunk (0x{self.HEADER_CHUNK_ID:X}) found in {self.filepath}")
        if self._cells is None:
            raise ValueError(f"No data chunk ({self.DATA_CHUNK_ID}) found in {self.filepath}")

    def _parse_header(self, data: bytes):
        """Parse header chunk."""
        if len(data) < 44:  # 4 + 4 + 4 + 16 + 16
            raise ValueError(f"Header chunk too small: {len(data)} bytes")

        version, level_vertex_count, game_vertex_count = struct.unpack_from('<III', data, 0)
        level_guid = data[12:28]
        game_guid = data[28:44]

        self._header = CrossTableHeader(
            version=version,
            level_vertex_count=level_vertex_count,
            game_vertex_count=game_vertex_count,
            level_guid=level_guid,
            game_guid=game_guid
        )

    def _parse_cells(self, data: bytes):
        """Parse cells chunk using numpy for efficiency."""
        dtype = np.dtype([('game_vertex_id', np.uint16), ('distance', np.float32)])
        self._cells = np.frombuffer(data, dtype=dtype)

        if self._header and len(self._cells) != self._header.level_vertex_count:
            raise ValueError(
                f"Cell count mismatch: {len(self._cells)} vs header's {self._header.level_vertex_count}"
            )

    @property
    def header(self) -> CrossTableHeader:
        """Get cross table header."""
        if self._header is None:
            raise RuntimeError("Cross table not loaded")
        return self._header

    @property
    def level_vertex_count(self) -> int:
        """Number of level vertices in this cross table."""
        return self.header.level_vertex_count

    @property
    def game_vertex_count(self) -> int:
        """Number of game vertices this level maps to."""
        return self.header.game_vertex_count

    def get_game_vertex_id(self, level_vertex_id: int) -> int:
        """
        Get the game vertex ID for a level vertex.

        Args:
            level_vertex_id: Level-local vertex ID

        Returns:
            Game graph vertex ID (local to this level, add offset for global)
        """
        if self._cells is None:
            raise RuntimeError("Cross table not loaded")
        if level_vertex_id < 0 or level_vertex_id >= len(self._cells):
            raise IndexError(f"Level vertex ID {level_vertex_id} out of range [0, {len(self._cells)})")
        return int(self._cells[level_vertex_id]['game_vertex_id'])

    def get_distance(self, level_vertex_id: int) -> float:
        """
        Get the distance from a level vertex to its game vertex.

        Args:
            level_vertex_id: Level-local vertex ID

        Returns:
            Distance in meters
        """
        if self._cells is None:
            raise RuntimeError("Cross table not loaded")
        if level_vertex_id < 0 or level_vertex_id >= len(self._cells):
            raise IndexError(f"Level vertex ID {level_vertex_id} out of range [0, {len(self._cells)})")
        return float(self._cells[level_vertex_id]['distance'])

    def find_level_vertices_for_game_vertex(self, game_vertex_id: int) -> np.ndarray:
        """
        Find all level vertices that map to a specific game vertex.

        Args:
            game_vertex_id: Local game vertex ID to search for

        Returns:
            Array of level vertex IDs
        """
        if self._cells is None:
            raise RuntimeError("Cross table not loaded")
        return np.where(self._cells['game_vertex_id'] == game_vertex_id)[0]

    def get_all_cells(self) -> np.ndarray:
        """
        Get all cross table cells as a numpy structured array.

        Returns:
            Structured array with 'game_vertex_id' and 'distance' fields
        """
        if self._cells is None:
            raise RuntimeError("Cross table not loaded")
        return self._cells


def find_game_vertex_from_cross_table(level_vertex_id: int, cross_table_path: Path) -> int:
    """
    Legacy function: Look up local game vertex ID from cross table.

    This is a compatibility wrapper for code that hasn't migrated to CrossTableParser.

    Args:
        level_vertex_id: Level-local vertex ID
        cross_table_path: Path to cross table file

    Returns:
        Local game vertex ID, or 0xFFFF on error
    """
    try:
        parser = CrossTableParser(cross_table_path)
        if level_vertex_id >= parser.level_vertex_count:
            return 0xFFFF
        return parser.get_game_vertex_id(level_vertex_id)
    except Exception:
        return 0xFFFF
