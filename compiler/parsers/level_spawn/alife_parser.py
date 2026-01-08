"""
ALife object parser for spawn entities.

Parses CSE_ALifeObject data from spawn packets.
"""

import struct
from typing import Optional

from .data_types import ALifeObject, SpawnEntity
from .string_utils import read_stringz_cp1251


def parse_alife_object(entity: SpawnEntity) -> Optional[ALifeObject]:
    """
    Parse CSE_ALifeObject data from a SpawnEntity's spawn_packet.

    This is used for entities that inherit from CSE_ALifeObject (most game entities).

    Args:
        entity: SpawnEntity to parse alife_object from

    Returns:
        ALifeObject if successfully parsed, None otherwise
    """
    try:
        spawn_packet = entity.spawn_packet
        version = entity.version

        # Navigate to the STATE data section (same as parse_entity_shapes)
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
        offset += 1

        # Entity name
        while offset < len(spawn_packet) and spawn_packet[offset] != 0:
            offset += 1
        offset += 1

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
        state_data = spawn_packet[offset:offset + data_size]
        state_offset = 0

        # Parse CSE_ALifeObject::STATE_Read
        alife = ALifeObject()

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

            if state_offset + 6 > len(state_data):
                return None
            alife.game_graph_id = struct.unpack_from('<H', state_data, state_offset)[0]
            state_offset += 2
            alife.distance = struct.unpack_from('<f', state_data, state_offset)[0]
            state_offset += 4

        if version >= 4:
            if state_offset + 4 > len(state_data):
                return alife
            dwDummy = struct.unpack_from('<I', state_data, state_offset)[0]
            alife.direct_control = bool(dwDummy)
            state_offset += 4

        if version >= 8:
            if state_offset + 4 > len(state_data):
                return alife
            alife.node_id = struct.unpack_from('<I', state_data, state_offset)[0]
            state_offset += 4

        if (version > 22) and (version <= 79):
            state_offset += 2  # spawn_id_old

        if (version > 23) and (version < 84):
            # spawn_control_old stringZ
            _, state_offset = read_stringz_cp1251(state_data, state_offset)

        if version > 49:
            if state_offset + 4 > len(state_data):
                return alife
            alife.object_flags = struct.unpack_from('<I', state_data, state_offset)[0]
            state_offset += 4

        if version > 57:
            if state_offset >= len(state_data):
                return alife
            alife.ini_string, state_offset = read_stringz_cp1251(state_data, state_offset)

        if version > 61:
            if state_offset + 4 > len(state_data):
                return alife
            alife.story_id = struct.unpack_from('<I', state_data, state_offset)[0]
            state_offset += 4

        if version > 111:
            if state_offset + 4 > len(state_data):
                return alife
            alife.spawn_story_id = struct.unpack_from('<I', state_data, state_offset)[0]
            state_offset += 4

        return alife

    except Exception:
        return None
