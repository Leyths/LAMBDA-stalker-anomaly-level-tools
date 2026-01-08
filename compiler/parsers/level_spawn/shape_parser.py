"""
Shape parser for spawn entities.

Parses CSE_Shape data from space_restrictor and similar entity types.
"""

import struct
from typing import List

from .data_types import Shape, SpawnEntity


def parse_entity_shapes(entity: SpawnEntity) -> List[Shape]:
    """
    Parse shape data from a SpawnEntity's spawn_packet.

    This is used for space_restrictor type entities that contain CSE_Shape data.
    The shapes define collision/trigger volumes relative to the entity position.

    Args:
        entity: SpawnEntity to parse shapes from

    Returns:
        List of Shape objects, or empty list if no shapes or not a shape-containing entity
    """
    # Only space_restrictor and similar types have shapes
    shape_sections = ['space_restrictor', 'script_zone', 'script_restr', 'camp_zone',
                      'smart_terrain', 'smart_zone']
    if entity.section_name not in shape_sections:
        return []

    try:
        spawn_packet = entity.spawn_packet
        version = entity.version

        # Navigate to the STATE data section
        # We need to re-parse the header to find the data_size offset
        offset = 0

        # Size prefix (if present)
        if len(spawn_packet) >= 2:
            potential_size = struct.unpack_from('<H', spawn_packet, 0)[0]
            if potential_size == len(spawn_packet) - 2:
                offset = 2

        # M_SPAWN type
        offset += 2

        # Section name
        while offset < len(spawn_packet) and spawn_packet[offset] != 0:
            offset += 1
        offset += 1  # Skip null terminator

        # Entity name
        while offset < len(spawn_packet) and spawn_packet[offset] != 0:
            offset += 1
        offset += 1  # Skip null terminator

        # gameid, rp
        offset += 2

        # Position (12 bytes)
        offset += 12

        # Angle (12 bytes)
        offset += 12

        # respawn, id, parent, phantom
        offset += 8

        # s_flags
        s_flags = struct.unpack_from('<H', spawn_packet, offset)[0]
        offset += 2

        # Version
        if s_flags & 0x20:  # M_SPAWN_VERSION
            offset += 2

        # Skip version-dependent fields
        if version > 120:
            offset += 2  # game_type
        if version > 69:
            offset += 2  # script_version
        if version > 70:
            client_data_size = struct.unpack_from('<H', spawn_packet, offset)[0]
            offset += 2 + client_data_size

        # spawn_id
        if version > 79:
            offset += 2

        # data_size
        data_size = struct.unpack_from('<H', spawn_packet, offset)[0]
        offset += 2

        # Now we're at the STATE data start
        state_start = offset
        state_data = spawn_packet[state_start:state_start + data_size]

        # Parse CSE_ALifeObject data first (we need to skip past it to get to shapes)
        state_offset = 0

        # CSE_ALifeObject::STATE_Read
        if version >= 1:
            if version > 24:
                if version < 83:
                    state_offset += 4  # spawn_probability float
            else:
                state_offset += 1  # old spawn_probability u8

            if version < 83:
                state_offset += 4  # dummy u32

            if version < 4:
                state_offset += 2  # wDummy u16

            state_offset += 2  # game_graph_id
            state_offset += 4  # distance

        if version >= 4:
            state_offset += 4  # direct_control u32

        if version >= 8:
            state_offset += 4  # node_id u32

        if (version > 22) and (version <= 79):
            state_offset += 2  # spawn_id_old

        if (version > 23) and (version < 84):
            # spawn_control_old stringZ
            while state_offset < len(state_data) and state_data[state_offset] != 0:
                state_offset += 1
            state_offset += 1

        if version > 49:
            state_offset += 4  # object_flags u32

        if version > 57:
            # ini_string stringZ
            while state_offset < len(state_data) and state_data[state_offset] != 0:
                state_offset += 1
            state_offset += 1

        if version > 61:
            state_offset += 4  # story_id u32

        if version > 111:
            state_offset += 4  # spawn_story_id u32

        # Now we're at the CSE_Shape data (cform_read)
        if state_offset >= len(state_data):
            return []

        shape_count = state_data[state_offset]
        state_offset += 1

        shapes = []
        for _ in range(shape_count):
            if state_offset >= len(state_data):
                break

            shape_type = state_data[state_offset]
            state_offset += 1

            if shape_type == 0:  # Sphere
                if state_offset + 16 > len(state_data):
                    break
                center = struct.unpack_from('<3f', state_data, state_offset)
                state_offset += 12
                radius = struct.unpack_from('<f', state_data, state_offset)[0]
                state_offset += 4
                shapes.append(Shape(
                    shape_type=0,
                    center=center,
                    radius=radius
                ))

            elif shape_type == 1:  # Box
                # Box format: 3x3 rotation matrix (9 floats) + translation (3 floats) = 48 bytes
                if state_offset + 48 > len(state_data):
                    break
                # Read 3x3 matrix as 3 axis vectors (each scaled by half-extent)
                axes = []
                for _ in range(3):
                    axis = struct.unpack_from('<3f', state_data, state_offset)
                    state_offset += 12
                    axes.append(axis)
                # Read translation
                translation = struct.unpack_from('<3f', state_data, state_offset)
                state_offset += 12
                shapes.append(Shape(
                    shape_type=1,
                    box_axes=axes,
                    box_translation=translation
                ))
            else:
                # Unknown shape type, stop parsing
                break

        return shapes

    except Exception:
        return []
