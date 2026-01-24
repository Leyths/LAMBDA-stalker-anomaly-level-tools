#!/usr/bin/env python3
"""
Spawn Entity GVID Remapper

Updates game_vertex_id and level_vertex_id for spawn entities based on position.
Also updates dest_game_vertex_id for level_changer entities.

Uses the central GameGraph object for all GVID resolution.
Supports config-driven level changer destination overrides.
"""

import struct
from typing import Dict, Optional, TYPE_CHECKING

from utils import logDebug, logError

from parsers import (
    parse_level_changer_data,
    parse_spawn_packet,
    SpawnEntity,
    find_game_vertex_from_cross_table,
)

if TYPE_CHECKING:
    from graph import GameGraph
    from .level_changer_config import LevelChangerConfig

# Sections that don't have CSE_ALifeObject graph IDs
SECTIONS_WITHOUT_GRAPH_IDS = {'graph_point'}


def _resolve_level_changer_destination(entity: SpawnEntity,
                                        entity_index: int,
                                        name_to_gvid: Dict[str, int],
                                        level_name: str = "",
                                        lc_config: Optional['LevelChangerConfig'] = None,
                                        game_graph: Optional['GameGraph'] = None) -> bytes:
    """
    Resolve level_changer destination from config using position-based GVID resolution.

    Config is authoritative - specifies destination level, position, and direction.
    GVID and LVID are computed from position using spatial lookup.

    Packet field layout (from dest_gvid_offset):
        +0:  dest_game_vertex_id (2 bytes, u16)
        +2:  dest_level_vertex_id (4 bytes, i32)
        +6:  dest_position (12 bytes, 3 floats)
        +18: dest_direction (12 bytes, 3 floats)

    Args:
        entity: SpawnEntity to resolve destination for
        entity_index: Entity index for logging
        name_to_gvid: Graph point name to GVID mapping (unused, kept for API compatibility)
        level_name: Source level name (for config lookup)
        lc_config: LevelChangerConfig (authoritative source)
        game_graph: GameGraph for position-based GVID/LVID lookup
    """
    spawn_packet = entity.spawn_packet

    try:
        # Parse packet structure to get the offset where we write destination data
        lc_data = parse_level_changer_data(entity)
        if lc_data is None:
            logError(f"[{entity_index}] level_changer '{entity.entity_name}': Failed to parse packet structure")
            return spawn_packet

        dest_gvid_offset = lc_data.dest_gvid_offset
        if dest_gvid_offset < 0:
            logError(f"[{entity_index}] level_changer '{entity.entity_name}': Invalid GVID offset")
            return spawn_packet

        # Config is authoritative - must exist for level changer to be processed
        if not lc_config or not level_name:
            return spawn_packet

        override = lc_config.get_override(level_name, entity.entity_name)
        if not override:
            # No config entry - this level changer should be filtered out by builder.py
            return spawn_packet

        dest_level_name = override.dest_level_name
        position = override.position
        direction = override.direction

        if not game_graph:
            logError(f"[{entity_index}] level_changer '{entity.entity_name}': No GameGraph for position lookup")
            return spawn_packet

        # Resolve GVID from position using spatial lookup
        new_dest_gvid = game_graph.get_gvid_for_position(dest_level_name, position)
        if new_dest_gvid is None:
            logError(f"[{entity_index}] level_changer '{entity.entity_name}': Cannot resolve position {position} on '{dest_level_name}' to GVID")
            return spawn_packet

        # Resolve LVID from position
        new_dest_lvid = game_graph.get_level_vertex_for_position(dest_level_name, position)
        if new_dest_lvid is None:
            logError(f"[{entity_index}] level_changer '{entity.entity_name}': Cannot resolve position {position} on '{dest_level_name}' to LVID")
            return spawn_packet

        # Update packet with all destination fields
        updated_packet = bytearray(spawn_packet)

        # Write GVID (2 bytes at offset +0)
        struct.pack_into('<H', updated_packet, dest_gvid_offset, new_dest_gvid)

        # Write LVID (4 bytes at offset +2)
        struct.pack_into('<i', updated_packet, dest_gvid_offset + 2, new_dest_lvid)

        # Write position (12 bytes at offset +6)
        struct.pack_into('<3f', updated_packet, dest_gvid_offset + 6, *position)

        # Write direction (12 bytes at offset +18)
        struct.pack_into('<3f', updated_packet, dest_gvid_offset + 18, *direction)

        logDebug(f"[{entity_index}] level_changer '{entity.entity_name}': {dest_level_name} pos={position} -> GVID {new_dest_gvid}, LVID {new_dest_lvid}")
        return bytes(updated_packet)

    except Exception as e:
        logError(f"[{entity_index}] level_changer '{entity.entity_name}': Failed to resolve destination: {e}")
        import traceback
        traceback.print_exc()
        return spawn_packet


def remap_entity_gvids(spawn_packet: bytes,
                       game_graph: 'GameGraph',
                       level_name: str,
                       entity_index: int = 0,
                       name_to_gvid: Optional[Dict[str, int]] = None,
                       lc_config: Optional['LevelChangerConfig'] = None) -> bytes:
    """
    Parse entity spawn packet and update graph IDs based on position.

    Uses the GameGraph object for all GVID resolution, which provides:
    - Cached level.ai data for position lookups
    - Cached cross table data for level->game vertex mapping
    - Automatic level offset calculation for global GVIDs

    For level_changer entities, also updates dest_game_vertex_id based on
    the dest_graph_point name. Supports config-driven destination overrides.

    Args:
        spawn_packet: The entity's spawn packet data (with size prefix)
        game_graph: GameGraph object for GVID resolution
        level_name: Name of the level this entity is on
        entity_index: Entity index for logging
        name_to_gvid: Graph point name to GVID mapping (for level_changer resolution)
        lc_config: Optional LevelChangerConfig for destination overrides

    Returns:
        Updated spawn packet (or original if update not needed/possible)
    """
    try:
        # Parse the spawn packet using the shared parser
        entity = parse_spawn_packet(spawn_packet, update_packet=None, has_size_prefix=True)
        if entity is None:
            return spawn_packet

        section_name = entity.section_name
        entity_name = entity.entity_name
        position = entity.position

        # Skip sections that don't have graph IDs
        if section_name in SECTIONS_WITHOUT_GRAPH_IDS:
            return spawn_packet

        # Verify we have valid offsets for updating
        if entity.game_vertex_id_offset < 0 or entity.level_vertex_id_offset < 0:
            return spawn_packet

        old_game_id = entity.game_vertex_id
        old_level_id = entity.level_vertex_id

        # Get paths from GameGraph for resolution
        # We use the same functions as the original resolver for consistency
        level_config = game_graph._get_level_config(level_name)
        if level_config is None:
            logError(f"[{entity_index}] {section_name} '{entity_name}': No level config for {level_name}")
            return spawn_packet

        level_ai = game_graph.get_level_ai_for_level(level_name)
        if level_ai is None:
            logError(f"[{entity_index}] {section_name} '{entity_name}': No level AI for {level_name}")
            return spawn_packet
        new_level_id = level_ai.find_nearest_vertex(position)
        if new_level_id is None:
            logError(f"[{entity_index}] {section_name} '{entity_name}': Could not find level vertex for position {position}")
            return spawn_packet

        cross_table_path = game_graph.cross_table_dir / f"{level_name}.gct"

        # Calculate new game_vertex_id from cross table + offset (using same function as original)
        local_game_id = find_game_vertex_from_cross_table(new_level_id, cross_table_path)
        if local_game_id == 0xFFFF:
            logError(f"[{entity_index}] {section_name} '{entity_name}': Could not find game vertex for level vertex {new_level_id}")
            return spawn_packet

        game_vertex_offset = game_graph.get_level_offset(level_name)
        new_game_id = local_game_id + game_vertex_offset

        # Check if update needed for base graph IDs
        if new_game_id == old_game_id and new_level_id == old_level_id:
            # Still need to check level_changer destinations even if base IDs unchanged
            if section_name == 'level_changer' and lc_config:
                return _resolve_level_changer_destination(entity, entity_index, name_to_gvid,
                                                          level_name, lc_config, game_graph)
            return spawn_packet

        # Update packet at the offsets provided by the parser
        updated_packet = bytearray(spawn_packet)
        struct.pack_into('<H', updated_packet, entity.game_vertex_id_offset, new_game_id)
        struct.pack_into('<I', updated_packet, entity.level_vertex_id_offset, new_level_id)

        debug_this = (entity_index % 1000 == 0)
        if debug_this:
            offset = game_graph.get_level_offset(level_name)
            local_game_id = new_game_id - offset
            logDebug(f"[{entity_index}] {section_name} '{entity_name}':")
            logDebug(f"  Position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")
            logDebug(f"  Level vertex: {old_level_id} -> {new_level_id}")
            logDebug(f"  Game vertex: {old_game_id} -> {new_game_id} (local={local_game_id} + offset={offset})")

        result_packet = bytes(updated_packet)

        # For level_changer entities, also resolve dest_game_vertex_id
        # Note: We need to re-parse with updated packet since offsets may have shifted
        if section_name == 'level_changer' and lc_config:
            updated_entity = parse_spawn_packet(result_packet, update_packet=None, has_size_prefix=True)
            if updated_entity:
                result_packet = _resolve_level_changer_destination(updated_entity, entity_index, name_to_gvid,
                                                                    level_name, lc_config, game_graph)

        return result_packet

    except Exception as e:
        logError(f"[{entity_index}]: {e}")
        import traceback
        traceback.print_exc()
        return spawn_packet
