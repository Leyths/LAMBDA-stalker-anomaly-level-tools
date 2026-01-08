#!/usr/bin/env python3
"""
Game Graph Serializer

Serializes merged game graph data to binary format for the all.spawn file.
Implements the file format from xr_graph_merge.cpp and game_graph.h
"""

import struct
from pathlib import Path
from typing import List, TYPE_CHECKING

from utils import log, logDebug, generate_guid

if TYPE_CHECKING:
    from graph import GameGraph

# Constants for binary format sizes
VERTEX_SIZE = 42  # bytes per CVertex
EDGE_SIZE = 6     # bytes per CEdge
DEATH_POINT_SIZE = 20  # bytes per CLevelPoint


class GameGraphSerializer:
    """
    Serialize game graph to binary format.

    Output structure (from xr_graph_merge.cpp lines 633-665):
    - CHeader
    - CVertex array
    - CEdge array
    - CLevelPoint array (death points)
    - Cross tables (per level)
    """

    def __init__(self, game_graph: 'GameGraph'):
        """
        Initialize serializer.

        Args:
            game_graph: GameGraph object containing all data
        """
        self.game_graph = game_graph

        # Compute GUID if not already set
        if game_graph.graph_guid is None:
            level_names = ",".join(sorted(l.name for l in game_graph.levels))
            game_graph.graph_guid = generate_guid(
                "game_graph",
                len(game_graph.vertices),
                len(game_graph.edges),
                len(game_graph.death_points),
                len(game_graph.levels),
                level_names
            )
        self.guid = game_graph.graph_guid

    def serialize(self) -> bytes:
        """
        Serialize complete game graph to bytes.

        Returns:
            Binary game graph data
        """
        import io

        # Calculate byte offsets before serializing
        self._calculate_offsets()

        buffer = io.BytesIO()

        self._serialize_header(buffer)
        self._serialize_vertices(buffer)
        self._serialize_edges(buffer)
        self._serialize_death_points(buffer)
        self._serialize_cross_tables(buffer)

        return buffer.getvalue()

    def _calculate_offsets(self):
        """
        Calculate edge and death point offsets for serialization.

        CRITICAL: Offsets are BYTE offsets from m_nodes (start of vertex array),
        NOT item counts. From game_graph_inline.h:

            (const CEdge *)((BYTE *)m_nodes + vertex(id)->edge_offset())

        So the structure is:
            [vertices: N * 42 bytes]
            [edges: M * 6 bytes]
            [death_points: P * 20 bytes]

        And edge_offset = vertices_size + cumulative_edge_bytes
        And death_point_offset = vertices_size + edges_size + cumulative_death_point_bytes
        """
        log("  Calculating offsets...")

        vertices = self.game_graph.vertices
        edges = self.game_graph.edges
        death_points = self.game_graph.death_points

        # Calculate base offsets
        vertices_size = len(vertices) * VERTEX_SIZE
        edges_size = len(edges) * EDGE_SIZE

        edges_base_offset = vertices_size
        death_points_base_offset = vertices_size + edges_size

        # Calculate per-vertex offsets
        cumulative_edge_bytes = 0
        cumulative_death_point_bytes = 0

        for vertex in vertices:
            # Set byte offsets from m_nodes
            vertex.edge_offset = edges_base_offset + cumulative_edge_bytes
            vertex.death_point_offset = death_points_base_offset + cumulative_death_point_bytes

            # Update cumulative byte counts
            cumulative_edge_bytes += len(vertex.edges) * EDGE_SIZE
            cumulative_death_point_bytes += len(vertex.death_points) * DEATH_POINT_SIZE

        logDebug(f"    Vertices: {len(vertices)} x {VERTEX_SIZE} = {vertices_size:,} bytes")
        logDebug(f"    Edges: {len(edges)} x {EDGE_SIZE} = {edges_size:,} bytes")
        logDebug(f"    Death points: {len(death_points)} x {DEATH_POINT_SIZE} = {len(death_points) * DEATH_POINT_SIZE:,} bytes")
        logDebug(f"    Edge base offset: {edges_base_offset:,}")
        logDebug(f"    Death point base offset: {death_points_base_offset:,}")

    def _serialize_header(self, f):
        """Serialize CHeader structure."""
        log("  Serializing header...")

        vertices = self.game_graph.vertices
        edges = self.game_graph.edges
        death_points = self.game_graph.death_points
        levels = self.game_graph.levels

        # Version
        version = 10
        f.write(struct.pack('<B', version))

        # Counts
        f.write(struct.pack('<H', len(vertices)))
        f.write(struct.pack('<I', len(edges)))
        f.write(struct.pack('<I', len(death_points)))

        # GUID
        f.write(self.guid)

        # Level count
        f.write(struct.pack('<B', len(levels)))

        # Write each level
        for level in levels:
            self._serialize_level_info(f, level)

    def _serialize_level_info(self, f, level):
        """
        Serialize SLevel structure.

        Structure from game_graph_inline.h SLevel::save():
        1. name (stringZ)
        2. offset (vec3 = 12 bytes)
        3. id (u8 = 1 byte)
        4. section (stringZ)
        5. guid (16 bytes)
        """
        # Name (null-terminated)
        name_bytes = level.name.encode('utf-8') + b'\x00'
        f.write(name_bytes)

        # Offset (3 floats = 12 bytes)
        f.write(struct.pack('<3f', *level.offset))

        # ID (u8 = 1 byte)
        f.write(struct.pack('<B', level.id))

        # Section (null-terminated)
        section_bytes = level.section.encode('utf-8') + b'\x00'
        f.write(section_bytes)

        # GUID (from level_guids dict, or zeros if not available)
        level_guid = self.game_graph.level_guids.get(level.id, b'\x00' * 16)
        f.write(level_guid)

    def _serialize_vertices(self, f):
        """Serialize CVertex array (42 bytes each)."""
        log("  Serializing vertices...")

        for vertex in self.game_graph.vertices:
            # Local point (12 bytes)
            f.write(struct.pack('<3f', *vertex.local_point))

            # Global point (12 bytes)
            f.write(struct.pack('<3f', *vertex.global_point))

            # Packed: level_id (8 bits) | level_vertex_id (24 bits)
            packed = (vertex.level_id & 0xFF) | ((vertex.level_vertex_id & 0xFFFFFF) << 8)
            f.write(struct.pack('<I', packed))

            # Vertex types (4 bytes)
            f.write(vertex.vertex_types)

            # Offsets and counts
            f.write(struct.pack('<I', vertex.edge_offset))
            f.write(struct.pack('<I', vertex.death_point_offset))
            f.write(struct.pack('<B', vertex.edge_count))
            f.write(struct.pack('<B', vertex.death_point_count))

    def _serialize_edges(self, f):
        """Serialize CEdge array (6 bytes each)."""
        log("  Serializing edges...")

        for edge in self.game_graph.edges:
            f.write(struct.pack('<H', edge.vertex_id))
            f.write(struct.pack('<f', edge.distance))

    def _serialize_death_points(self, f):
        """Serialize CLevelPoint array (20 bytes each)."""
        log("  Serializing death points...")

        for dp in self.game_graph.death_points:
            f.write(struct.pack('<3f', *dp.position))
            f.write(struct.pack('<I', dp.level_vertex_id))
            f.write(struct.pack('<f', dp.distance))

    def _serialize_cross_tables(self, f):
        """Serialize embedded cross tables."""
        log("  Serializing cross tables...")

        # Access cross_tables directly from game_graph to avoid stale reference
        cross_tables = self.game_graph.cross_tables
        for ct in cross_tables:
            f.write(ct.data)

        log(f"    Serialized {len(cross_tables)} cross tables")
