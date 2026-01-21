"""
M_UPDATE packet creation and handling.

Creates and manipulates M_UPDATE packets for spawn entities based on their section type.
"""

import struct
import io

from utils import logDebug, logWarning
from parsers import parse_spawn_packet
from extraction import extract_section_name

from .entity_categories import (
    W_GL_SECTIONS, SHOTGUN_SECTIONS, SIMPLE_WEAPON_SECTIONS,
    OUTFIT_SECTIONS, HELMET_SECTIONS, ST_MGUN_SECTIONS,
    PHYSICS_SECTIONS, PHYSICS_PREFIXES, STATIC_SECTIONS, STATIC_PREFIXES,
    is_monster_section, is_physics_section, is_static_section,
)


def get_expected_update_size(section: str) -> int:
    """
    Get expected M_UPDATE packet size (including 2-byte size prefix) for a section.
    Returns None if size varies or is unknown.

    Based on create_update_packet logic.
    """
    # Actor has complex variable-length data
    if section == 'actor':
        return None  # Variable

    # Ammo: size(2) + msg_type(2) + num_items(1) + ammo_left(2) = 7
    if section.startswith('ammo_') or section == 'wpn_rpg7_missile':
        return 7

    # Outfits and helmets: size(2) + msg_type(2) + num_items(1) + condition(1) = 6
    if section in OUTFIT_SECTIONS or section in HELMET_SECTIONS:
        return 6

    # Stationary mgun: size(2) + msg_type(2) + working(1) + direction(12) = 17
    if section in ST_MGUN_SECTIONS:
        return 17

    # Weapons have variable sizes based on type
    if section.startswith('wpn_') and not any(x in section for x in ['addon_', 'fake_missile']):
        return None  # Variable based on weapon type

    # Physics objects: size(2) + msg_type(2) + num_items(1) = 5
    if is_physics_section(section):
        return 5

    # Static objects: just size prefix (2) + message type (2) = 4 bytes
    if is_static_section(section):
        return 4  # size(2) + msg_type(2)

    # Generic items: size(2) + msg_type(2) + num_items(1) = 5
    return 5


def remap_update_packet_gvids(update_packet: bytes, section_name: str,
                               spawn_packet: bytes) -> bytes:
    """
    Remap m_tNextGraphID and m_tPrevGraphID in M_UPDATE packets for monsters/stalkers.

    CSE_ALifeMonsterAbstract::UPDATE_Write binary layout (56 bytes)
    Used by engine to sync monster state. We must remap graph vertex fields:
      [0-43]  Header, health, position, rotation, team info
      [44-45] u16: m_tNextGraphID  <- REMAP: target vertex in AI navigation
      [46-47] u16: m_tPrevGraphID  <- REMAP: previous vertex in AI navigation
      [48-55] Path distance fields

    The engine initializes these as: m_tNextGraphID = m_tPrevGraphID = m_tGraphID
    We remap monster/stalker GVIDs to the entity's current resolved position since
    the old values may reference locations that no longer exist or are in wrong
    positions in the reorganized game graph.

    Args:
        update_packet: The M_UPDATE packet with size prefix
        section_name: Entity section name
        spawn_packet: The resolved spawn packet (to get current GVID)

    Returns:
        Updated packet, or original if no changes needed
    """
    # Only process monster/stalker sections
    if not is_monster_section(section_name):
        return update_packet

    # Monster M_UPDATE packets are 56 bytes (size prefix + msg_type + creature data + monster data)
    if len(update_packet) < 56:
        return update_packet

    # Offsets for GVIDs in monster M_UPDATE packet
    NEXT_GVID_OFFSET = 44
    PREV_GVID_OFFSET = 46

    try:
        # Read current GVIDs from M_UPDATE
        old_next_gvid = struct.unpack_from('<H', update_packet, NEXT_GVID_OFFSET)[0]
        old_prev_gvid = struct.unpack_from('<H', update_packet, PREV_GVID_OFFSET)[0]

        # Get the resolved GVID from spawn packet using parser
        entity = parse_spawn_packet(spawn_packet)
        if entity is None or entity.game_vertex_id == 0xFFFF:
            return update_packet

        resolved_gvid = entity.game_vertex_id

        # Update the packet
        updated = bytearray(update_packet)
        struct.pack_into('<H', updated, NEXT_GVID_OFFSET, resolved_gvid)
        struct.pack_into('<H', updated, PREV_GVID_OFFSET, resolved_gvid)

        logDebug(f"    Remapped M_UPDATE GVIDs for '{entity.entity_name}' ({section_name}): "
                 f"next {old_next_gvid}->{resolved_gvid}, prev {old_prev_gvid}->{resolved_gvid}")

        return bytes(updated)

    except Exception as e:
        logWarning(f"Failed to remap M_UPDATE GVIDs for {section_name}: {e}")
        return update_packet


def create_update_packet(spawn_packet: bytes) -> bytes:
    """
    Create M_UPDATE packet.
    Fixed for CoP:
    - Removed door_lab_x8, device_pda from Static (Needs 1 byte).
    - Added handlers for Traders (12 bytes) and Stationary MG (13 bytes).
    """
    buffer = io.BytesIO()
    section = extract_section_name(spawn_packet)

    # Start with M_UPDATE message type (2 bytes)
    buffer.write(struct.pack('<H', 0))

    # --- 1. ACTOR ---
    if section == 'actor':
        buffer.write(struct.pack('<f', 1.0))  # fHealth
        buffer.write(struct.pack('<I', 0))  # timestamp
        buffer.write(struct.pack('<B', 0))  # flags
        buffer.write(struct.pack('<fff', 0.0, 0.0, 0.0))  # o_Position
        buffer.write(struct.pack('<f', 0.0))  # o_model
        buffer.write(struct.pack('<f', 0.0))  # o_torso.yaw
        buffer.write(struct.pack('<f', 0.0))  # o_torso.pitch
        buffer.write(struct.pack('<f', 0.0))  # o_torso.roll
        buffer.write(struct.pack('<B', 0))  # s_team
        buffer.write(struct.pack('<B', 0))  # s_squad
        buffer.write(struct.pack('<B', 0))  # s_group
        buffer.write(struct.pack('<H', 0))  # mstate
        buffer.write(struct.pack('<Hf', 0, 0.0))  # accel
        buffer.write(struct.pack('<Hf', 0, 0.0))  # velocity
        buffer.write(struct.pack('<f', 0.0))  # fRadiation
        buffer.write(struct.pack('<B', 0))  # weapon
        buffer.write(struct.pack('<H', 0))  # m_u16NumItems

    # --- 2. AMMO ---
    elif section.startswith('ammo_') or section == 'wpn_rpg7_missile':
        buffer.write(struct.pack('<B', 0))  # num_items
        buffer.write(struct.pack('<H', 30))  # ammo_left

    # --- 3. WEAPONS ---
    elif section.startswith('wpn_') and not any(x in section for x in ['addon_', 'fake_missile']):
        is_w_gl = section in W_GL_SECTIONS
        is_shotgun = section in SHOTGUN_SECTIONS
        is_simple = section in SIMPLE_WEAPON_SECTIONS

        if is_w_gl:
            buffer.write(struct.pack('<B', 0))

        buffer.write(struct.pack('<B', 0))  # num_items
        buffer.write(struct.pack('<B', 255))  # condition
        buffer.write(struct.pack('<B', 0))  # weapon_flags
        buffer.write(struct.pack('<H', 0))  # ammo_elapsed
        buffer.write(struct.pack('<B', 0))  # addon_flags
        buffer.write(struct.pack('<B', 0))  # ammo_type
        buffer.write(struct.pack('<B', 0))  # weapon_state
        buffer.write(struct.pack('<B', 0))  # zoom

        if not is_simple:
            buffer.write(struct.pack('<B', 0))  # current_fire_mode

        if is_shotgun:
            buffer.write(struct.pack('<B', 0))  # ammo_ids list count

    # --- 4. OUTFITS & HELMETS ---
    elif section in OUTFIT_SECTIONS or section in HELMET_SECTIONS:
        buffer.write(struct.pack('<B', 0))  # num_items
        buffer.write(struct.pack('<B', 255))  # condition

    # --- 5. STATIONARY MGUNS ---
    elif section in ST_MGUN_SECTIONS:
        buffer.write(struct.pack('<B', 0))  # working
        buffer.write(struct.pack('<fff', 0.0, 0.0, 0.0))  # dest_enemy_direction

    # --- 6. PHYSICS OBJECTS (Write num_items=0 for ACDC compatibility) ---
    # These are cse_alife_object_physic classes that need the num_items byte
    elif is_physics_section(section):
        buffer.write(struct.pack('<B', 0))  # num_items = 0 (no physics state)

    # --- 7. STATIC OBJECTS (Write 0 Bytes) ---
    elif is_static_section(section):
        pass  # Write NOTHING

    # --- 8. GENERIC ITEMS (Write 1 Byte) ---
    # Catches device_pda, misc items, etc.
    else:
        buffer.write(struct.pack('<B', 0))  # num_items = 0

    packet_data = buffer.getvalue()
    final_buffer = io.BytesIO()
    final_buffer.write(struct.pack('<H', len(packet_data)))
    final_buffer.write(packet_data)

    return final_buffer.getvalue()
