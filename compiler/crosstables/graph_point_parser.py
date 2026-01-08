"""
Graph Point Parser

Extracts graph points from binary spawn files.
Handles both NEW format (editor output) and OLD format (all.spawn extraction).
"""

import struct
from pathlib import Path
from typing import List, Dict, Optional

from utils import log, logDebug
from parsers import read_stringz, detect_spawn_format
from .data_types import GraphPoint


def extract_graph_points_from_binary(spawn_path: Path) -> Dict[str, GraphPoint]:
    """
    Extract graph points directly from binary spawn file.
    Handles both formats:
    - NEW format: Each chunk contains M_SPAWN packet directly
    - OLD format: Nested chunks (from all.spawn extraction via extract_spawns.py)

    Returns dict keyed by name for easy merging.
    """
    graph_points = {}

    if not spawn_path.exists():
        return graph_points

    with open(spawn_path, 'rb') as f:
        data = f.read()

    # Detect format by checking first chunk
    spawn_format = detect_spawn_format(data)
    logDebug(f"  Detected format: {spawn_format}")

    if spawn_format == 'NEW':
        # NEW format: Direct M_SPAWN packets in each chunk
        offset = 0
        while offset < len(data) - 8:
            chunk_id, chunk_size = struct.unpack_from('<II', data, offset)
            chunk_data = data[offset + 8:offset + 8 + chunk_size]

            if len(chunk_data) != chunk_size:
                break

            gp = parse_graph_point_packet(chunk_data)
            if gp:
                graph_points[gp.name] = gp

            offset += 8 + chunk_size

    else:  # OLD format
        # OLD format: Nested chunk structure from extract_spawns.py
        # Structure: Outer chunk -> Sub-chunk 0 (id=0) -> u16 size prefix + M_SPAWN packet
        offset = 0
        while offset < len(data) - 8:
            chunk_id, chunk_size = struct.unpack_from('<II', data, offset)
            chunk_data = data[offset + 8:offset + 8 + chunk_size]

            if len(chunk_data) != chunk_size:
                break

            # Parse sub-chunks
            sub_offset = 0
            spawn_packet = None

            while sub_offset < len(chunk_data) - 8:
                sub_id, sub_size = struct.unpack_from('<II', chunk_data, sub_offset)
                sub_data = chunk_data[sub_offset + 8:sub_offset + 8 + sub_size]

                if sub_id == 0:  # M_SPAWN packet (with size prefix)
                    # Read size prefix
                    if len(sub_data) >= 2:
                        packet_size = struct.unpack_from('<H', sub_data, 0)[0]
                        # Extract M_SPAWN packet (skip size prefix)
                        spawn_packet = sub_data[2:2 + packet_size]
                    break

                sub_offset += 8 + sub_size

            # Parse the M_SPAWN packet
            if spawn_packet:
                gp = parse_graph_point_packet(spawn_packet)
                if gp:
                    graph_points[gp.name] = gp

            offset += 8 + chunk_size

    return graph_points


def parse_graph_point_packet(packet_data: bytes) -> Optional[GraphPoint]:
    """
    Parse an M_SPAWN packet and extract graph_point data.

    Handles two STATE data formats:
    - OLD format (spawn_extractor output): Full CSE_ALifeObject + CSE_ALifeGraphPoint STATE
    - NEW format (editor output): Only CSE_ALifeGraphPoint STATE (no vertex IDs assigned yet)

    Returns GraphPoint if it's a graph_point entity, None otherwise.
    """
    try:
        offset = 0

        # M_SPAWN type check
        if len(packet_data) < 2:
            return None
        msg_type = struct.unpack_from('<H', packet_data, offset)[0]
        if msg_type != 1:
            return None
        offset += 2

        # Section name
        section, offset = read_stringz(packet_data, offset)

        # Only process graph_points
        if section != 'graph_point':
            return None

        # Entity name
        name, offset = read_stringz(packet_data, offset)

        # Skip gameid, rp
        offset += 2

        # Position
        if offset + 12 > len(packet_data):
            return None
        position = struct.unpack_from('<3f', packet_data, offset)
        offset += 12

        # Skip angle
        offset += 12

        # Skip respawn, id, parent, phantom
        offset += 8

        # Flags
        if offset + 2 > len(packet_data):
            return None
        s_flags = struct.unpack_from('<H', packet_data, offset)[0]
        offset += 2

        # Version
        version = 0
        if s_flags & 0x20:
            if offset + 2 > len(packet_data):
                return None
            version = struct.unpack_from('<H', packet_data, offset)[0]
            offset += 2

        # Skip version-dependent fields
        if version > 120:
            offset += 2
        if version > 69:
            offset += 2
        if version > 70:
            if offset + 2 > len(packet_data):
                return None
            client_size = struct.unpack_from('<H', packet_data, offset)[0]
            offset += 2 + client_size
        if version > 79:
            offset += 2

        # Data size - size of STATE data
        if offset + 2 > len(packet_data):
            return None
        data_size = struct.unpack_from('<H', packet_data, offset)[0]
        offset += 2

        state_data_start = offset
        state_data = packet_data[offset:offset + data_size]

        # Determine format by content, not size:
        # OLD format: CSE_ALifeObject fields followed by CSE_ALifeGraphPoint
        # NEW format: Only CSE_ALifeGraphPoint (connection strings + location types)
        #
        # Detection: OLD format starts with binary values (game_vertex_id as u16),
        # NEW format starts with connection_point_name string (printable ASCII or null)

        level_vertex_id = -1
        connection_point_name = ""
        connection_level_name = ""
        locations = bytes([0, 0, 0, 0])

        # Detect format by checking if first bytes look like a string or binary data
        is_new_format = True
        if data_size >= 20 and len(state_data) >= 2:
            first_byte = state_data[0]
            # Binary data typically has non-printable first byte (game_vertex_id low byte)
            # String data starts with printable char or null terminator
            if first_byte != 0 and (first_byte < 0x20 or first_byte > 0x7E):
                is_new_format = False

        if not is_new_format:
            # OLD format: Parse CSE_ALifeObject STATE + CSE_ALifeGraphPoint STATE
            state_offset = 0

            # game_vertex_id (u16)
            if state_offset + 2 > len(state_data):
                return None
            state_offset += 2

            # distance (f32), direct_control (u32)
            state_offset += 8

            # level_vertex_id (u32)
            if state_offset + 4 > len(state_data):
                return None
            level_vertex_id = struct.unpack_from('<I', state_data, state_offset)[0]
            state_offset += 4

            # flags (u32)
            state_offset += 4

            # custom_data (stringZ)
            if state_offset < len(state_data):
                _, state_offset = read_stringz(state_data, state_offset)

            # story_id, spawn_story_id (u32, u32)
            state_offset += 8

            # CSE_ALifeGraphPoint STATE
            if state_offset < len(state_data):
                connection_point_name, state_offset = read_stringz(state_data, state_offset)
            if state_offset < len(state_data):
                connection_level_name, state_offset = read_stringz(state_data, state_offset)
            if state_offset + 4 <= len(state_data):
                locations = state_data[state_offset:state_offset + 4]

        else:
            # NEW format: Only CSE_ALifeGraphPoint STATE (no vertex IDs assigned yet)
            state_offset = 0

            if state_offset < len(state_data):
                connection_point_name, state_offset = read_stringz(state_data, state_offset)
            if state_offset < len(state_data):
                connection_level_name, state_offset = read_stringz(state_data, state_offset)
            if state_offset + 4 <= len(state_data):
                locations = state_data[state_offset:state_offset + 4]

            # level_vertex_id will be assigned later from level.ai
            level_vertex_id = -1

        return GraphPoint(
            index=0,  # Will be assigned later
            name=name,
            position=position,
            level_vertex_id=level_vertex_id,
            location_types=locations,
            connection_point_name=connection_point_name,
            connection_level_name=connection_level_name
        )

    except Exception as e:
        return None


def extract_and_merge_graph_points(level_spawn_path: Path,
                                   original_spawn_path: Optional[Path] = None) -> List[GraphPoint]:
    """
    Extract and merge graph points from level.spawn and optional original_spawn.

    Priority logic:
    - NEW level.spawn names take priority (they're more descriptive: "mar_graph_point_0214")
    - OLD original spawn has generic names ("graph_point_1094") but has level_vertex_id values

    Merging strategy:
    1. Load all NEW graph points (descriptive names, level_vertex_id=-1)
    2. For each OLD graph point, find if there's a matching NEW graph point by position
    3. If matched: Copy level_vertex_id from OLD to NEW (keep NEW name)
    4. If not matched: Add OLD graph point as new entry

    This ensures: descriptive names from NEW, level_vertex_ids from OLD, no position duplicates.
    """
    # Load from NEW level.spawn FIRST - these names have priority
    graph_points: List[GraphPoint] = []
    new_count = 0
    if level_spawn_path.exists():
        new_gps = extract_graph_points_from_binary(level_spawn_path)
        new_count = len(new_gps)
        graph_points = list(new_gps.values())
        logDebug(f"  Loaded {new_count} graph points from level.spawn")

    # Load from original spawn SECOND - match by position and fill level_vertex_id
    if original_spawn_path and original_spawn_path.exists():
        original_gps = extract_graph_points_from_binary(original_spawn_path)
        original_count = len(original_gps)

        lvid_filled = 0
        unmatched_count = 0

        for old_name, old_gp in original_gps.items():
            # Try to find matching NEW graph point by position (within 1 meter tolerance)
            matched = False
            for new_gp in graph_points:
                dx = new_gp.position[0] - old_gp.position[0]
                dy = new_gp.position[1] - old_gp.position[1]
                dz = new_gp.position[2] - old_gp.position[2]
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5

                if dist < 1.0:  # Same position (within 1m)
                    # Found match - copy level_vertex_id from OLD to NEW
                    if new_gp.level_vertex_id == -1 and old_gp.level_vertex_id != -1:
                        new_gp.level_vertex_id = old_gp.level_vertex_id
                        lvid_filled += 1
                    matched = True
                    break

            # If no match found, add OLD graph point as new entry
            if not matched:
                graph_points.append(old_gp)
                unmatched_count += 1

        logDebug(f"  Loaded {original_count} graph points from original spawn:")
        logDebug(f"    - Matched by position & filled level_vertex_id: {lvid_filled}")
        logDebug(f"    - Unmatched (added as new): {unmatched_count}")

    # Assign indices
    for i, gp in enumerate(graph_points):
        gp.index = i

    log(f"  Total merged: {len(graph_points)} graph points")
    return graph_points
