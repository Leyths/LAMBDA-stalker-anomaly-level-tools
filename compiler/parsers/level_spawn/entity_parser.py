"""
Entity parser for spawn packets.

Parses M_SPAWN packets into SpawnEntity objects.
"""

import struct
from typing import Optional

from ..base import read_stringz
from .data_types import SpawnEntity


# Message type constant
M_SPAWN = 1


def parse_spawn_packet(spawn_packet: bytes, update_packet: Optional[bytes] = None,
                       has_size_prefix: bool = True) -> Optional[SpawnEntity]:
    """
    Parse a spawn packet into a SpawnEntity.

    This is a shared utility function used by both LevelSpawnParser (for level.spawn)
    and AllSpawnSpawnIterator (for all.spawn).

    Args:
        spawn_packet: Raw spawn packet bytes
        update_packet: Optional M_UPDATE packet bytes
        has_size_prefix: Whether packet starts with 2-byte size prefix

    Returns:
        SpawnEntity if successfully parsed, None otherwise
    """
    try:
        offset = 0

        # Size prefix (optional)
        if has_size_prefix:
            if len(spawn_packet) < 2:
                return None
            offset += 2

        # Check for size prefix in packets that claim not to have one
        # Some packets have it embedded, detect by checking if first word matches length
        if not has_size_prefix and len(spawn_packet) >= 4:
            potential_size = struct.unpack_from('<H', spawn_packet, 0)[0]
            if potential_size == len(spawn_packet) - 2:
                offset = 2

        # M_SPAWN type
        if offset + 2 > len(spawn_packet):
            return None
        msg_type = struct.unpack_from('<H', spawn_packet, offset)[0]
        if msg_type != M_SPAWN:
            return None
        offset += 2

        # Section name
        section_name, offset = read_stringz(spawn_packet, offset)

        # Entity name
        entity_name, offset = read_stringz(spawn_packet, offset)

        # gameid, rp
        offset += 2

        # Position
        if offset + 12 > len(spawn_packet):
            return None
        position = struct.unpack_from('<3f', spawn_packet, offset)
        offset += 12

        # Angle
        if offset + 12 > len(spawn_packet):
            return None
        angle = struct.unpack_from('<3f', spawn_packet, offset)
        offset += 12

        # respawn, id, parent, phantom
        offset += 8

        # s_flags
        if offset + 2 > len(spawn_packet):
            return None
        s_flags = struct.unpack_from('<H', spawn_packet, offset)[0]
        offset += 2

        # Version
        version = 0
        if s_flags & 0x20:  # M_SPAWN_VERSION
            if offset + 2 > len(spawn_packet):
                return None
            version = struct.unpack_from('<H', spawn_packet, offset)[0]
            offset += 2

        # Skip version-dependent fields
        if version > 120:
            offset += 2  # game_type
        if version > 69:
            offset += 2  # script_version
        if version > 70:
            if offset + 2 > len(spawn_packet):
                return None
            client_data_size = struct.unpack_from('<H', spawn_packet, offset)[0]
            offset += 2 + client_data_size

        # spawn_id
        spawn_id = 0
        if version > 79:
            if offset + 2 > len(spawn_packet):
                return None
            spawn_id = struct.unpack_from('<H', spawn_packet, offset)[0]
            offset += 2

        # data_size
        if offset + 2 > len(spawn_packet):
            return None
        data_size = struct.unpack_from('<H', spawn_packet, offset)[0]
        offset += 2

        # Parse STATE data for graph IDs
        # CSE_ALifeObject STATE layout: game_vertex_id(2) + distance(4) + direct_control(4) + level_vertex_id(4) + ...
        game_vertex_id = 0xFFFF
        level_vertex_id = 0xFFFFFFFF
        game_vertex_id_offset = -1
        level_vertex_id_offset = -1

        if section_name != 'graph_point' and offset + 14 <= len(spawn_packet):
            game_vertex_id_offset = offset
            level_vertex_id_offset = offset + 10  # After game_vertex_id(2) + distance(4) + direct_control(4)
            game_vertex_id = struct.unpack_from('<H', spawn_packet, game_vertex_id_offset)[0]
            level_vertex_id = struct.unpack_from('<I', spawn_packet, level_vertex_id_offset)[0]

        return SpawnEntity(
            section_name=section_name,
            entity_name=entity_name,
            position=position,
            angle=angle,
            game_vertex_id=game_vertex_id,
            level_vertex_id=level_vertex_id,
            spawn_packet=spawn_packet,
            update_packet=update_packet,
            spawn_id=spawn_id,
            version=version,
            game_vertex_id_offset=game_vertex_id_offset,
            level_vertex_id_offset=level_vertex_id_offset
        )

    except Exception:
        return None
