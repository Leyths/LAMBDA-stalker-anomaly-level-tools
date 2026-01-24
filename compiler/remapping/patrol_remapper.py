#!/usr/bin/env python3
"""
Patrol Path GVID Remapper

Updates game_vertex_id and level_vertex_id for patrol path points.
Uses the central GameGraph object for all GVID resolution.

Also handles validation of patrol point positions and filtering of invalid points.
"""

import struct
import io
import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from utils import logDebug, logError, log

from parsers import read_stringz, find_nearest_level_vertex, find_game_vertex_from_cross_table

if TYPE_CHECKING:
    from graph import GameGraph


def parse_patrol_point(data: bytes, offset: int) -> Tuple[dict, int]:
    """
    Parse a CPatrolPoint from binary data

    Returns:
        (point_dict, new_offset)
    """
    point = {}

    # Name
    point['name'], offset = read_stringz(data, offset)

    # Position
    if offset + 12 > len(data):
        return None, offset
    point['position'] = struct.unpack_from('<3f', data, offset)
    offset += 12

    # Flags
    if offset + 4 > len(data):
        return None, offset
    point['flags'] = struct.unpack_from('<I', data, offset)[0]
    offset += 4

    # Level vertex ID
    if offset + 4 > len(data):
        return None, offset
    point['level_vertex_id'] = struct.unpack_from('<I', data, offset)[0]
    offset += 4

    # Game vertex ID
    if offset + 2 > len(data):
        return None, offset
    point['game_vertex_id'] = struct.unpack_from('<H', data, offset)[0]
    offset += 2

    return point, offset


def is_position_in_bounds(position: Tuple[float, float, float],
                          min_bounds: Tuple[float, float, float],
                          max_bounds: Tuple[float, float, float],
                          tolerance: float = 5.0) -> bool:
    """
    Check if a position is within level bounds (with tolerance)

    Args:
        position: (x, y, z) position to check
        min_bounds: (min_x, min_y, min_z) level bounds
        max_bounds: (max_x, max_y, max_z) level bounds
        tolerance: Extra margin to allow outside strict bounds

    Returns:
        True if position is within bounds (+/- tolerance)
    """
    x, y, z = position
    min_x, min_y, min_z = min_bounds
    max_x, max_y, max_z = max_bounds

    return (min_x - tolerance <= x <= max_x + tolerance and
            min_y - tolerance <= y <= max_y + tolerance and
            min_z - tolerance <= z <= max_z + tolerance)


def remap_patrol_gvids(patrol_data: bytes,
                       game_graph: 'GameGraph',
                       level_name: str) -> bytes:
    """
    Update game_vertex_id and level_vertex_id for all points in a patrol path

    Uses GameGraph for GVID resolution.

    Args:
        patrol_data: Binary CPatrolPath data
        game_graph: GameGraph object for GVID resolution
        level_name: Name of the level

    Returns:
        Updated patrol_data with corrected IDs
    """
    # Get level AI navigator for this level
    level_ai = game_graph.get_level_ai_for_level(level_name)
    if level_ai is None:
        return patrol_data

    # Parse the patrol structure
    offset = 0
    chunks = []

    while offset < len(patrol_data):
        if offset + 8 > len(patrol_data):
            break

        chunk_id = struct.unpack_from('<I', patrol_data, offset)[0]
        chunk_size = struct.unpack_from('<I', patrol_data, offset + 4)[0]
        offset += 8

        if offset + chunk_size > len(patrol_data):
            break

        chunk_data = patrol_data[offset:offset + chunk_size]
        offset += chunk_size

        # Update chunk 1 (vertices with CPatrolPoint data)
        if chunk_id == 1:
            chunk_data = _update_vertices_chunk(chunk_data, game_graph, level_name)

        chunks.append((chunk_id, chunk_data))

    # Rebuild patrol data
    buffer = io.BytesIO()
    for chunk_id, chunk_data in chunks:
        buffer.write(struct.pack('<I', chunk_id))
        buffer.write(struct.pack('<I', len(chunk_data)))
        buffer.write(chunk_data)

    return buffer.getvalue()


def _update_vertices_chunk(vertices_data: bytes,
                           game_graph: 'GameGraph',
                           level_name: str) -> bytes:
    """Update all vertex sub-chunks to fix graph IDs"""
    offset = 0
    updated_vertices = []

    while offset < len(vertices_data):
        if offset + 8 > len(vertices_data):
            break

        v_chunk_id = struct.unpack_from('<I', vertices_data, offset)[0]
        v_chunk_size = struct.unpack_from('<I', vertices_data, offset + 4)[0]
        offset += 8

        if offset + v_chunk_size > len(vertices_data):
            break

        v_chunk_data = vertices_data[offset:offset + v_chunk_size]
        offset += v_chunk_size

        # Update this vertex's sub-chunks
        updated_v_data = _update_vertex_subchunks(v_chunk_data, game_graph, level_name)
        updated_vertices.append((v_chunk_id, updated_v_data))

    # Rebuild vertices chunk
    buffer = io.BytesIO()
    for v_chunk_id, v_data in updated_vertices:
        buffer.write(struct.pack('<I', v_chunk_id))
        buffer.write(struct.pack('<I', len(v_data)))
        buffer.write(v_data)

    return buffer.getvalue()


def _update_vertex_subchunks(vertex_data: bytes,
                             game_graph: 'GameGraph',
                             level_name: str) -> bytes:
    """Update vertex ID and CPatrolPoint sub-chunks"""
    offset = 0
    subchunks = []

    while offset < len(vertex_data):
        if offset + 8 > len(vertex_data):
            break

        chunk_id = struct.unpack_from('<I', vertex_data, offset)[0]
        chunk_size = struct.unpack_from('<I', vertex_data, offset + 4)[0]
        offset += 8

        if offset + chunk_size > len(vertex_data):
            break

        chunk_data = vertex_data[offset:offset + chunk_size]
        offset += chunk_size

        # Update chunk 1 (CPatrolPoint)
        if chunk_id == 1:
            chunk_data = _update_patrol_point_data(chunk_data, game_graph, level_name)

        subchunks.append((chunk_id, chunk_data))

    # Rebuild vertex data
    buffer = io.BytesIO()
    for chunk_id, chunk_data in subchunks:
        buffer.write(struct.pack('<I', chunk_id))
        buffer.write(struct.pack('<I', len(chunk_data)))
        buffer.write(chunk_data)

    return buffer.getvalue()


def _update_patrol_point_data(point_data: bytes,
                              game_graph: 'GameGraph',
                              level_name: str) -> bytes:
    """Update graph IDs in a CPatrolPoint using GameGraph"""
    # Parse the point
    point, _ = parse_patrol_point(point_data, 0)
    if not point:
        return point_data

    position = point['position']

    # Get paths from GameGraph for resolution
    # We use the same functions as the original resolver for consistency
    level_config = game_graph._get_level_config(level_name)
    if level_config is None:
        return point_data

    level_ai_path = game_graph.base_path / level_config.path / "level.ai"
    cross_table_path = game_graph.cross_table_dir / f"{level_name}.gct"

    # Use same functions as original for position-based GVID resolution
    new_level_vertex_id = find_nearest_level_vertex(position, level_ai_path)
    if new_level_vertex_id == 0xFFFFFFFF:
        # Couldn't find vertex, keep original
        logError(f"Unable to find new level vertex ID for {level_name}: [{position[0]}{position[1]}{position[2]}]")
        return point_data

    local_game_id = find_game_vertex_from_cross_table(new_level_vertex_id, cross_table_path)
    if local_game_id == 0xFFFF:
        # Couldn't find game vertex, keep original
        logError(f"Unable to find new game vertex ID for {level_name}: [{position[0]}{position[1]}{position[2]}]")
        return point_data

    game_vertex_offset = game_graph.get_level_offset(level_name)
    new_game_vertex_id = local_game_id + game_vertex_offset

    # Rebuild the point data with updated IDs
    buffer = io.BytesIO()

    # Name
    buffer.write(point['name'].encode('utf-8') + b'\x00')

    # Position
    buffer.write(struct.pack('<3f', *position))

    # Flags
    buffer.write(struct.pack('<I', point['flags']))

    # NEW level_vertex_id
    buffer.write(struct.pack('<I', new_level_vertex_id))

    # NEW game_vertex_id
    buffer.write(struct.pack('<H', new_game_vertex_id))

    return buffer.getvalue()


def validate_and_remap_patrols(patrols: Dict[str, bytes],
                                game_graph: 'GameGraph',
                                level_name: str) -> Dict[str, bytes]:
    """
    Update and validate patrol paths, filtering out invalid points.

    This function:
    1. Gets level bounds from GameGraph's cached level AI
    2. Pre-loads level.ai positions and cross table into memory ONCE
    3. For each patrol path:
       - Validates each point's position is within level bounds
       - Updates level_vertex_id via cached lookup
       - Updates game_vertex_id via cached cross table lookup
       - Filters out points with invalid IDs
       - Re-indexes edges to account for removed points
       - Removes patrol paths where all points are invalid
    4. Returns dict of valid patrol paths only

    Args:
        patrols: Dict of {patrol_name: patrol_data}
        game_graph: GameGraph object for GVID resolution
        level_name: Name of the level

    Returns:
        Dict of {patrol_name: validated_patrol_data}
    """
    # Get level AI for bounds checking
    level_ai = game_graph.get_level_ai_for_level(level_name)
    if level_ai is None:
        logError(f"        ERROR: Could not get level AI for {level_name}, returning patrols unvalidated")
        return patrols

    # Get bounds from level AI header
    min_bounds = level_ai.header['min']
    max_bounds = level_ai.header['max']

    # Pre-load cross table data into memory ONCE
    cross_table_cache = game_graph.get_cross_table_cache(level_name)

    validated_patrols = {}
    total_points_filtered = 0
    total_patrols_removed = 0

    for name, patrol_data in patrols.items():
        result = _validate_and_update_patrol(
            name, patrol_data, game_graph, level_name, min_bounds, max_bounds,
            level_ai, cross_table_cache
        )

        if result is None:
            total_patrols_removed += 1
        else:
            validated_data, points_filtered = result
            total_points_filtered += points_filtered
            validated_patrols[name] = validated_data

    if total_points_filtered > 0:
        logDebug(f"        Filtered {total_points_filtered} out-of-bounds patrol points")
    if total_patrols_removed > 0:
        logDebug(f"        Removed {total_patrols_removed} patrol paths (all points invalid)")

    return validated_patrols


def _validate_and_update_patrol(name: str,
                                 patrol_data: bytes,
                                 game_graph: 'GameGraph',
                                 level_name: str,
                                 min_bounds: Tuple[float, float, float],
                                 max_bounds: Tuple[float, float, float],
                                 level_ai=None,
                                 cross_table_cache=None) -> Optional[Tuple[bytes, int]]:
    """
    Validate and update a single patrol path.

    Returns:
        Tuple of (updated_patrol_data, points_filtered_count) or None if all points invalid
    """
    # Parse patrol chunks
    offset = 0
    chunks = []

    while offset < len(patrol_data):
        if offset + 8 > len(patrol_data):
            break

        chunk_id = struct.unpack_from('<I', patrol_data, offset)[0]
        chunk_size = struct.unpack_from('<I', patrol_data, offset + 4)[0]
        offset += 8

        if offset + chunk_size > len(patrol_data):
            break

        chunk_data = patrol_data[offset:offset + chunk_size]
        offset += chunk_size
        chunks.append((chunk_id, chunk_data))

    # Find vertex count chunk (chunk 0), vertices chunk (chunk 1) and edges chunk (chunk 2)
    vertex_count_chunk_idx = None
    vertices_chunk_data = None
    edges_chunk_data = None
    vertices_chunk_idx = None
    edges_chunk_idx = None

    for i, (chunk_id, chunk_data) in enumerate(chunks):
        if chunk_id == 0:
            vertex_count_chunk_idx = i
        elif chunk_id == 1:
            vertices_chunk_data = chunk_data
            vertices_chunk_idx = i
        elif chunk_id == 2:
            edges_chunk_data = chunk_data
            edges_chunk_idx = i

    if vertices_chunk_data is None:
        # No vertices, just return original
        return patrol_data, 0

    # Parse and validate vertices
    valid_vertices, old_to_new_idx, points_filtered = _parse_and_validate_vertices(
        vertices_chunk_data, game_graph, level_name, min_bounds, max_bounds,
        level_ai, cross_table_cache
    )

    if not valid_vertices:
        # All points invalid, remove entire patrol
        return None

    # Update vertex count in chunk 0
    if vertex_count_chunk_idx is not None:
        chunks[vertex_count_chunk_idx] = (0, struct.pack('<I', len(valid_vertices)))

    # Rebuild vertices chunk
    new_vertices_data = _rebuild_vertices_chunk(valid_vertices)
    chunks[vertices_chunk_idx] = (1, new_vertices_data)

    # Update edges if they exist
    if edges_chunk_data is not None and old_to_new_idx:
        new_edges_data = _remap_edges(edges_chunk_data, old_to_new_idx)
        chunks[edges_chunk_idx] = (2, new_edges_data)

    # Rebuild patrol data
    buffer = io.BytesIO()
    for chunk_id, chunk_data in chunks:
        buffer.write(struct.pack('<I', chunk_id))
        buffer.write(struct.pack('<I', len(chunk_data)))
        buffer.write(chunk_data)

    return buffer.getvalue(), points_filtered


def _parse_and_validate_vertices(vertices_data: bytes,
                                  game_graph: 'GameGraph',
                                  level_name: str,
                                  min_bounds: Tuple[float, float, float],
                                  max_bounds: Tuple[float, float, float],
                                  level_ai=None,
                                  cross_table_cache=None) -> Tuple[List[Tuple[int, bytes]], Dict[int, int], int]:
    """
    Parse vertices, validate positions, update IDs, filter invalid.

    Returns:
        Tuple of (valid_vertices, old_to_new_index_map, points_filtered_count)
        Note: old_to_new_index_map maps OLD vertex_id (from sub-chunk 0) to NEW sequential index
    """
    offset = 0
    valid_vertices = []
    old_to_new_idx = {}
    points_filtered = 0

    while offset < len(vertices_data):
        if offset + 8 > len(vertices_data):
            break

        v_chunk_id = struct.unpack_from('<I', vertices_data, offset)[0]
        v_chunk_size = struct.unpack_from('<I', vertices_data, offset + 4)[0]
        offset += 8

        if offset + v_chunk_size > len(vertices_data):
            break

        v_chunk_data = vertices_data[offset:offset + v_chunk_size]
        offset += v_chunk_size

        # Parse the vertex's sub-chunks to get position, vertex_id, and update IDs
        is_valid, updated_data, old_vertex_id = _validate_and_update_vertex(
            v_chunk_data, game_graph, level_name, min_bounds, max_bounds,
            level_ai, cross_table_cache
        )

        if is_valid:
            new_idx = len(valid_vertices)
            # Map OLD vertex_id (from sub-chunk 0) to NEW sequential index
            # This is what edges reference
            old_to_new_idx[old_vertex_id] = new_idx
            # Update the vertex data with the new vertex_id
            updated_data = _update_vertex_id_in_data(updated_data, new_idx)
            valid_vertices.append((new_idx, updated_data))
        else:
            points_filtered += 1

    return valid_vertices, old_to_new_idx, points_filtered


def _update_vertex_id_in_data(vertex_data: bytes, new_vertex_id: int) -> bytes:
    """Update the vertex_id in sub-chunk 0 of vertex data."""
    # Parse sub-chunks
    offset = 0
    subchunks = []

    while offset < len(vertex_data):
        if offset + 8 > len(vertex_data):
            break

        chunk_id = struct.unpack_from('<I', vertex_data, offset)[0]
        chunk_size = struct.unpack_from('<I', vertex_data, offset + 4)[0]
        offset += 8

        if offset + chunk_size > len(vertex_data):
            break

        chunk_data = vertex_data[offset:offset + chunk_size]
        offset += chunk_size

        # Update sub-chunk 0 (vertex_id) with new value
        if chunk_id == 0:
            chunk_data = struct.pack('<I', new_vertex_id)

        subchunks.append((chunk_id, chunk_data))

    # Rebuild vertex data
    buffer = io.BytesIO()
    for chunk_id, chunk_data in subchunks:
        buffer.write(struct.pack('<I', chunk_id))
        buffer.write(struct.pack('<I', len(chunk_data)))
        buffer.write(chunk_data)

    return buffer.getvalue()


def _validate_and_update_vertex(vertex_data: bytes,
                                 game_graph: 'GameGraph',
                                 level_name: str,
                                 min_bounds: Tuple[float, float, float],
                                 max_bounds: Tuple[float, float, float],
                                 level_ai=None,
                                 cross_table_cache=None) -> Tuple[bool, bytes, int]:
    """
    Validate a vertex position and update its IDs.

    Returns:
        Tuple of (is_valid, updated_vertex_data, old_vertex_id)
        old_vertex_id is the original vertex_id from sub-chunk 0 (used for edge remapping)
    """
    # Parse sub-chunks
    offset = 0
    subchunks = []
    point_data = None
    point_chunk_idx = None
    old_vertex_id = 0  # Will be read from sub-chunk 0

    while offset < len(vertex_data):
        if offset + 8 > len(vertex_data):
            break

        chunk_id = struct.unpack_from('<I', vertex_data, offset)[0]
        chunk_size = struct.unpack_from('<I', vertex_data, offset + 4)[0]
        offset += 8

        if offset + chunk_size > len(vertex_data):
            break

        chunk_data = vertex_data[offset:offset + chunk_size]
        offset += chunk_size

        if chunk_id == 0 and chunk_size >= 4:  # vertex_id sub-chunk
            old_vertex_id = struct.unpack_from('<I', chunk_data, 0)[0]
        elif chunk_id == 1:  # CPatrolPoint
            point_data = chunk_data
            point_chunk_idx = len(subchunks)

        subchunks.append((chunk_id, chunk_data))

    if point_data is None:
        # No point data, can't validate
        return True, vertex_data, old_vertex_id

    # Parse the point
    point, _ = parse_patrol_point(point_data, 0)
    if not point:
        return True, vertex_data, old_vertex_id

    position = point['position']

    # Check if position is in bounds
    if not is_position_in_bounds(position, min_bounds, max_bounds):
        return False, vertex_data, old_vertex_id

    # Use cached data for fast lookups if available
    if level_ai is not None and cross_table_cache is not None:
        # Find nearest vertex using grid-based spatial index (O(1) average)
        new_level_vertex_id = level_ai.find_nearest_vertex(position)
        if new_level_vertex_id is None or new_level_vertex_id == 0xFFFFFFFF:
            return False, vertex_data, old_vertex_id

        # Look up game vertex from cached cross table
        local_game_id = cross_table_cache.get_game_vertex(new_level_vertex_id)
        if local_game_id == 0xFFFF:
            return False, vertex_data, old_vertex_id
    else:
        # Fallback to file-based lookups
        level_config = game_graph._get_level_config(level_name)
        if level_config is None:
            return False, vertex_data, old_vertex_id

        level_ai_path = game_graph.base_path / level_config.path / "level.ai"
        cross_table_path = game_graph.cross_table_dir / f"{level_name}.gct"

        new_level_vertex_id = find_nearest_level_vertex(position, level_ai_path)
        if new_level_vertex_id == 0xFFFFFFFF:
            return False, vertex_data, old_vertex_id

        local_game_id = find_game_vertex_from_cross_table(new_level_vertex_id, cross_table_path)
        if local_game_id == 0xFFFF:
            return False, vertex_data, old_vertex_id

    game_vertex_offset = game_graph.get_level_offset(level_name)
    new_game_vertex_id = local_game_id + game_vertex_offset

    # Rebuild point data with updated IDs
    point_buffer = io.BytesIO()
    point_buffer.write(point['name'].encode('utf-8') + b'\x00')
    point_buffer.write(struct.pack('<3f', *position))
    point_buffer.write(struct.pack('<I', point['flags']))
    point_buffer.write(struct.pack('<I', new_level_vertex_id))
    point_buffer.write(struct.pack('<H', new_game_vertex_id))

    subchunks[point_chunk_idx] = (1, point_buffer.getvalue())

    # Rebuild vertex data
    buffer = io.BytesIO()
    for chunk_id, chunk_data in subchunks:
        buffer.write(struct.pack('<I', chunk_id))
        buffer.write(struct.pack('<I', len(chunk_data)))
        buffer.write(chunk_data)

    return True, buffer.getvalue(), old_vertex_id


def _rebuild_vertices_chunk(valid_vertices: List[Tuple[int, bytes]]) -> bytes:
    """Rebuild vertices chunk with re-indexed vertex IDs"""
    buffer = io.BytesIO()
    for new_idx, vertex_data in valid_vertices:
        buffer.write(struct.pack('<I', new_idx))
        buffer.write(struct.pack('<I', len(vertex_data)))
        buffer.write(vertex_data)
    return buffer.getvalue()


def _remap_edges(edges_data: bytes, old_to_new_idx: Dict[int, int]) -> bytes:
    """
    Remap edge vertex indices after filtering.
    Remove edges that reference removed vertices.

    Edges chunk format (flat, NOT chunked):
    - For each source vertex with outgoing edges:
      - source_vertex_id (u32)
      - edge_count (u32)
      - For each edge: target_vertex_id (u32), weight (f32)
    """
    offset = 0
    remapped_edges = {}  # new_source_id -> [(new_target_id, weight), ...]

    while offset + 8 <= len(edges_data):
        source_vertex_id = struct.unpack_from('<I', edges_data, offset)[0]
        offset += 4

        edge_count = struct.unpack_from('<I', edges_data, offset)[0]
        offset += 4

        # Check if source vertex was kept
        if source_vertex_id not in old_to_new_idx:
            # Source vertex was removed, skip all its edges
            offset += edge_count * 8  # 8 bytes per edge (u32 + f32)
            continue

        new_source_id = old_to_new_idx[source_vertex_id]

        # Process edges from this source
        for _ in range(edge_count):
            if offset + 8 > len(edges_data):
                break

            target_vertex_id = struct.unpack_from('<I', edges_data, offset)[0]
            weight = struct.unpack_from('<f', edges_data, offset + 4)[0]
            offset += 8

            # Check if target vertex was kept
            if target_vertex_id not in old_to_new_idx:
                # Target vertex was removed, skip this edge
                continue

            new_target_id = old_to_new_idx[target_vertex_id]

            # Add to remapped edges
            if new_source_id not in remapped_edges:
                remapped_edges[new_source_id] = []
            remapped_edges[new_source_id].append((new_target_id, weight))

    # Rebuild edges chunk in flat format
    buffer = io.BytesIO()
    for source_id in sorted(remapped_edges.keys()):
        edges = remapped_edges[source_id]
        buffer.write(struct.pack('<I', source_id))  # source vertex id
        buffer.write(struct.pack('<I', len(edges)))  # edge count
        for target_id, weight in edges:
            buffer.write(struct.pack('<I', target_id))  # target vertex id
            buffer.write(struct.pack('<f', weight))  # weight

    return buffer.getvalue()


