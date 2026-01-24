#!/usr/bin/env python3
"""
Cross Table Remapper

Reads per-level .gct cross table files and remaps local game vertex IDs
to global IDs for embedding in the merged game graph.
"""

import struct
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

from utils import log, logWarning

if TYPE_CHECKING:
    from levels import LevelsConfig


@dataclass
class RemappedCrossTable:
    """Cross table data with remapped global vertex IDs"""
    level_name: str
    level_id: int
    data: bytes  # Ready for embedding in game graph


class CrossTableRemapper:
    """
    Reads cross tables from .gct files and remaps vertex IDs to global space.

    The per-level .gct files contain local game vertex IDs (0-based for each level).
    When merging into the unified game graph, these must be remapped to global IDs.
    """

    def __init__(self, levels_config: 'LevelsConfig', vertices, graph_guid: bytes):
        """
        Initialize remapper.

        Args:
            levels_config: Level configurations (provides paths and lookups)
            vertices: Merged game vertices (to calculate offsets and totals)
            graph_guid: GUID of the merged game graph
        """
        self.levels_config = levels_config
        self.vertices = vertices
        self.graph_guid = graph_guid
        self.total_game_vertices = len(vertices)

        # Calculate vertex offset for each level (cumulative count)
        self._calculate_level_offsets()

    def _calculate_level_offsets(self):
        """Calculate the starting game vertex ID offset for each level."""
        self.level_offsets: Dict[int, int] = {}
        cumulative_offset = 0

        # Sort levels by ID to ensure consistent ordering
        sorted_levels = sorted(self.levels_config.levels, key=lambda l: l.id)

        for level in sorted_levels:
            self.level_offsets[level.id] = cumulative_offset
            # Count vertices for this level
            level_vertex_count = sum(1 for v in self.vertices if v.level_id == level.id)
            cumulative_offset += level_vertex_count

    def remap_all(self) -> List[RemappedCrossTable]:
        """
        Remap cross tables for all levels.

        Returns:
            List of RemappedCrossTable objects in level ID order
        """
        log("  Remapping cross tables...")

        results = []
        sorted_levels = sorted(self.levels_config.levels, key=lambda l: l.id)

        for level in sorted_levels:
            offset = self.level_offsets[level.id]
            data = self._remap_level(level, offset)

            if data is None:
                logWarning(f"No cross table for {level.name}, creating empty placeholder")
                data = self._create_empty_cross_table(level)

            results.append(RemappedCrossTable(
                level_name=level.name,
                level_id=level.id,
                data=data
            ))

        log(f"    Remapped {len(results)} cross tables")
        return results

    def _remap_level(self, level, vertex_offset: int) -> Optional[bytes]:
        """
        Read cross table from .gct file and remap game vertex IDs.

        Args:
            level: Level configuration
            vertex_offset: Offset to add to all game vertex IDs

        Returns:
            Binary cross table data ready for embedding, or None if not found
        """
        import io

        cross_table_path = self.levels_config.get_cross_table_path(level)

        if not cross_table_path or not cross_table_path.exists():
            logWarning(f"Cross table not found: {cross_table_path}")
            return None

        try:
            with open(cross_table_path, 'rb') as f:
                file_data = f.read()

            # Parse chunked format
            offset = 0
            header_data = None
            cells_data = None

            while offset < len(file_data):
                chunk_id = struct.unpack_from('<I', file_data, offset)[0]
                chunk_size = struct.unpack_from('<I', file_data, offset + 4)[0]
                offset += 8

                if chunk_id == 0xFFFF:  # Header chunk
                    header_data = file_data[offset:offset + chunk_size]
                elif chunk_id == 1:  # Data chunk
                    cells_data = file_data[offset:offset + chunk_size]

                offset += chunk_size

            if not header_data or not cells_data:
                logWarning(f"Invalid cross table format for {level.name}")
                return None

            # Parse header
            version, level_vertex_count, game_vertex_count, level_guid, game_guid = struct.unpack(
                '<III16s16s', header_data
            )

            # Remap cells: add vertex_offset to each game_vertex_id
            remapped_cells = bytearray()

            for i in range(level_vertex_count):
                cell_offset = i * 6
                old_game_vertex_id = struct.unpack_from('<H', cells_data, cell_offset)[0]
                distance = struct.unpack_from('<f', cells_data, cell_offset + 2)[0]

                # Apply offset to remap to global game vertex ID
                new_game_vertex_id = old_game_vertex_id + vertex_offset

                if new_game_vertex_id >= 0xFFFF:
                    logWarning(f"Remapped vertex ID overflow {level.name}: {new_game_vertex_id}")
                    new_game_vertex_id = 0xFFFF

                remapped_cells.extend(struct.pack('<H', new_game_vertex_id))
                remapped_cells.extend(struct.pack('<f', distance))

            # Build embedded cross table
            # Header: version, level_vertex_count, game_vertex_count (TOTAL), level_guid, game_guid
            embedded_header = struct.pack(
                '<III16s16s',
                version,
                level_vertex_count,
                self.total_game_vertices,
                level_guid,
                self.graph_guid
            )

            # Calculate total size (includes the size field itself)
            total_size = 4 + len(embedded_header) + len(remapped_cells)

            # Build final data
            buffer = io.BytesIO()
            buffer.write(struct.pack('<I', total_size))
            buffer.write(embedded_header)
            buffer.write(remapped_cells)

            return buffer.getvalue()

        except Exception as e:
            logWarning(f"Error reading cross table for {level.name}: {e}")
            return None

    def _create_empty_cross_table(self, level) -> bytes:
        """
        Create an empty/placeholder cross table for a level without one.
        """
        import io

        level_ai_path = self.levels_config.get_level_ai_path(level)

        level_vertex_count = 0
        level_guid = b'\x00' * 16

        if level_ai_path and level_ai_path.exists():
            try:
                with open(level_ai_path, 'rb') as f:
                    f.seek(4)  # Skip version
                    level_vertex_count = struct.unpack('<I', f.read(4))[0]
                    f.seek(40)  # Seek to GUID
                    level_guid = f.read(16)
            except Exception:
                pass

        if level_vertex_count == 0:
            # Minimal placeholder (header only)
            header = struct.pack(
                '<III16s16s',
                10,  # version
                0,   # level vertex count
                self.total_game_vertices,
                level_guid,
                self.graph_guid
            )
            total_size = 4 + len(header)

            buffer = io.BytesIO()
            buffer.write(struct.pack('<I', total_size))
            buffer.write(header)
            return buffer.getvalue()

        # Cross table pointing all vertices to game vertex 0 with max distance
        header = struct.pack(
            '<III16s16s',
            10,  # version
            level_vertex_count,
            self.total_game_vertices,
            level_guid,
            self.graph_guid
        )

        # All cells point to vertex 0 with infinite distance
        cells = struct.pack('<Hf', 0, 3.4028235e+38) * level_vertex_count

        total_size = 4 + len(header) + len(cells)

        buffer = io.BytesIO()
        buffer.write(struct.pack('<I', total_size))
        buffer.write(header)
        buffer.write(cells)

        return buffer.getvalue()
