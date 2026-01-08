"""
Level changer parser for spawn entities.

Parses CSE_ALifeLevelChanger destination data from spawn packets.
"""

import struct
from typing import Optional

from .data_types import LevelChangerData, SpawnEntity
from .string_utils import read_stringz_cp1251


def parse_level_changer_data(entity: SpawnEntity) -> Optional[LevelChangerData]:
    """
    Parse CSE_ALifeLevelChanger destination data from a SpawnEntity's spawn_packet.

    Parses FORWARD through the known packet structure to reach level_changer fields:

    Inheritance chain: CSE_ALifeLevelChanger -> CSE_ALifeSpaceRestrictor ->
                       CSE_ALifeDynamicObjectVisual (CSE_ALifeDynamicObject + CSE_Visual) + CSE_Shape

    Packet structure:
    - Size prefix (optional 2 bytes)
    - M_SPAWN header (2 bytes)
    - section_name, entity_name (stringZ)
    - CSE_Abstract data (gameid, rp, position, angle, etc.)
    - CSE_ALifeObject STATE data
    - CSE_Visual (visual_name stringZ)
    - CSE_Shape (variable size based on shape count and types)
    - CSE_ALifeSpaceRestrictor (restrictor_type u8)
    - CSE_ALifeLevelChanger (dest_gvid, dest_lvid, dest_pos, dest_dir, dest_level_name, dest_graph_point)

    Args:
        entity: SpawnEntity to parse level_changer data from

    Returns:
        LevelChangerData if successfully parsed, None otherwise
    """
    if entity.section_name != 'level_changer':
        return None

    return parse_level_changer_packet(entity.spawn_packet)


def parse_level_changer_packet(spawn_packet: bytes) -> Optional[LevelChangerData]:
    """
    Parse CSE_ALifeLevelChanger data from a raw spawn packet.

    This parses forward through the packet structure to find the level_changer fields.
    Can be used standalone without a SpawnEntity wrapper.

    Args:
        spawn_packet: Raw spawn packet bytes (with or without size prefix)

    Returns:
        LevelChangerData if successfully parsed, None otherwise
    """
    try:
        data = spawn_packet
        offset = 0

        # Size prefix (2 bytes) - check if present
        if len(data) >= 4:
            potential_size = struct.unpack_from('<H', data, 0)[0]
            if potential_size == len(data) - 2:
                offset = 2

        # M_SPAWN type (2 bytes)
        offset += 2

        # section_name (stringZ)
        section_name, offset = read_stringz_cp1251(data, offset)

        # Verify this is a level_changer
        if section_name != 'level_changer':
            return None

        # entity_name (stringZ)
        _, offset = read_stringz_cp1251(data, offset)

        # gameid (1), rp (1)
        offset += 2

        # position (vec3 = 12 bytes)
        offset += 12

        # angle (vec3 = 12 bytes)
        offset += 12

        # respawn_time (2), id (2), id_parent (2), id_phantom (2)
        offset += 8

        # s_flags (2 bytes)
        if offset + 2 > len(data):
            return None
        s_flags = struct.unpack_from('<H', data, offset)[0]
        offset += 2

        # version (optional)
        version = 0
        if s_flags & 0x20:  # M_SPAWN_VERSION
            if offset + 2 > len(data):
                return None
            version = struct.unpack_from('<H', data, offset)[0]
            offset += 2

        if version > 120:
            offset += 2  # game_type
        if version > 69:
            offset += 2  # script_version
        if version > 70:
            if offset + 2 > len(data):
                return None
            client_data_size = struct.unpack_from('<H', data, offset)[0]
            offset += 2 + client_data_size
        if version > 79:
            offset += 2  # spawn_id

        # data_size (2 bytes)
        if offset + 2 > len(data):
            return None
        offset += 2

        # === CSE_ALifeObject STATE data ===
        # game_vertex_id (2)
        offset += 2
        # distance (4)
        offset += 4
        # direct_control (4)
        offset += 4
        # level_vertex_id (4)
        offset += 4
        # s_flags (4)
        offset += 4
        # custom_data (stringZ)
        _, offset = read_stringz_cp1251(data, offset)
        # story_id (4)
        offset += 4
        # spawn_story_id (4)
        offset += 4

        # === CSE_ALifeDynamicObject === (nothing additional)
        # NOTE: CSE_ALifeSpaceRestrictor inherits from CSE_ALifeDynamicObject, NOT CSE_ALifeDynamicObjectVisual
        # So there is NO visual_name string here!

        # === CSE_Shape (cform_read) ===
        # shape count (1 byte)
        if offset >= len(data):
            return None
        shape_count = data[offset]
        offset += 1

        for _ in range(shape_count):
            if offset >= len(data):
                return None
            shape_type = data[offset]
            offset += 1
            if shape_type == 0:
                # Sphere: vec3 P (12) + float R (4) = 16 bytes
                offset += 16
            elif shape_type == 1:
                # Box: 3x3 matrix (36) + translation (12) = 48 bytes
                offset += 48
            else:
                # Unknown shape type - cannot continue parsing
                return None

        # === CSE_ALifeSpaceRestrictor ===
        # restrictor_type (1 byte)
        offset += 1

        # === CSE_ALifeLevelChanger ===
        # Now we're at the level_changer data
        if offset + 30 > len(data):  # Minimum: 2 + 4 + 12 + 12 = 30 bytes before strings
            return None

        # Record offset for dest_game_vertex_id (for in-place updates)
        dest_gvid_offset = offset

        # dest_game_vertex_id (2)
        dest_gvid = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        # dest_level_vertex_id (4)
        dest_lvid = struct.unpack_from('<i', data, offset)[0]  # signed int32
        offset += 4
        # dest_position (12)
        dest_pos = struct.unpack_from('<3f', data, offset)
        offset += 12
        # dest_direction (12)
        dest_dir = struct.unpack_from('<3f', data, offset)
        offset += 12
        # dest_level_name (stringZ)
        dest_level_name, offset = read_stringz_cp1251(data, offset)
        # dest_graph_point (stringZ)
        dest_graph_point, offset = read_stringz_cp1251(data, offset)

        # silent_mode (1 byte) - optional, may not be present in all versions
        silent_mode = False
        if offset < len(data):
            silent_mode = bool(data[offset])

        return LevelChangerData(
            dest_game_vertex_id=dest_gvid,
            dest_level_vertex_id=dest_lvid,
            dest_position=dest_pos,
            dest_direction=dest_dir,
            dest_level_name=dest_level_name,
            dest_graph_point=dest_graph_point,
            silent_mode=silent_mode,
            dest_gvid_offset=dest_gvid_offset
        )

    except Exception:
        return None
