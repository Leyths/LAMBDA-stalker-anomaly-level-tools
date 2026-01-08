#!/usr/bin/env python3
"""
Spawn Entity Extractor

Extracts and merges spawn entities from level.spawn files and old spawn data.
This is Phase 1 of the build pipeline - read-only extraction with no GVID remapping.

The extraction handles:
- Loading entities from level.spawn files
- Loading entities from old spawn files (with wrapper format detection)
- Merging entities to preserve custom_data/ini_string and M_UPDATE packets
- Blacklist filtering
"""

import struct
import fnmatch
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from utils import log


def load_blacklist(blacklist_path: Path) -> Tuple[Set[str], List[str]]:
    """
    Load spawn blacklist from ini file.

    Args:
        blacklist_path: Path to the blacklist ini file

    Returns:
        Tuple of (exact_names set, wildcard_patterns list)
    """
    exact_names = set()
    wildcard_patterns = []

    if not blacklist_path or not blacklist_path.exists():
        return exact_names, wildcard_patterns

    with open(blacklist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith(';'):
                continue
            if ';' in line:
                line = line.split(';')[0].strip()
            if '#' in line:
                line = line.split('#')[0].strip()
            if not line:
                continue
            if '*' in line:
                wildcard_patterns.append(line)
            else:
                exact_names.add(line)

    return exact_names, wildcard_patterns


def is_blacklisted(entity_name: str, section_name: str,
                   exact_names: Set[str], wildcard_patterns: List[str]) -> bool:
    """
    Check if an entity should be excluded based on blacklist.

    Args:
        entity_name: Name of the entity
        section_name: Section/class name of the entity
        exact_names: Set of exact names to blacklist
        wildcard_patterns: List of wildcard patterns to match

    Returns:
        True if entity should be excluded
    """
    if entity_name in exact_names or section_name in exact_names:
        return True
    for pattern in wildcard_patterns:
        if fnmatch.fnmatch(entity_name, pattern) or fnmatch.fnmatch(section_name, pattern):
            return True
    return False


def extract_entity_name(packet: bytes) -> str:
    """
    Extract entity name from spawn packet (with size prefix).

    Args:
        packet: Spawn packet bytes with u16 size prefix

    Returns:
        Entity name string, or empty string if extraction fails
    """
    try:
        offset = 2  # Skip size prefix
        offset += 2  # Skip M_SPAWN type
        # Skip section name
        end = packet.find(b'\x00', offset)
        if end == -1:
            return ""
        offset = end + 1
        # Read entity name
        end = packet.find(b'\x00', offset)
        if end == -1:
            return ""
        return packet[offset:end].decode('utf-8', errors='replace')
    except Exception:
        return ""


def extract_section_name(packet: bytes) -> str:
    """
    Extract section name from spawn packet (with size prefix).

    Args:
        packet: Spawn packet bytes with u16 size prefix

    Returns:
        Section name string, or empty string if extraction fails
    """
    try:
        offset = 2  # Skip size prefix
        offset += 2  # Skip M_SPAWN type
        # Read section name
        end = packet.find(b'\x00', offset)
        if end == -1:
            return ""
        return packet[offset:end].decode('utf-8', errors='replace')
    except Exception:
        return ""


def load_entities_from_spawn_file(spawn_path: Path) -> Dict[str, Tuple[bytes, Optional[bytes]]]:
    """
    Load entities from a spawn file.
    Detects if file has Wrapper chunks (Spawn+Update) or flat chunks.

    Args:
        spawn_path: Path to the spawn file

    Returns:
        Dict mapping entity_name to (spawn_packet, update_packet) tuple.
        update_packet may be None if not present.
    """
    entities = {}

    with open(spawn_path, 'rb') as f:
        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break

            chunk_id, chunk_size = struct.unpack('<II', chunk_header)
            chunk_data = f.read(chunk_size)

            if len(chunk_data) != chunk_size:
                break

            # === DETECT WRAPPER CHUNK ===
            is_wrapper = False
            spawn_packet = None
            update_packet = None

            if chunk_size > 8:
                # Check for Sub-chunk 0 header: ID=0 (4 bytes), Size (4 bytes)
                sub0_id, sub0_size = struct.unpack('<II', chunk_data[0:8])
                if sub0_id == 0 and (sub0_size + 8) <= chunk_size:
                    # Looks like a wrapper!
                    is_wrapper = True
                    # Extract Spawn Packet (it already has u16 size prefix inside the wrapper)
                    spawn_packet = chunk_data[8: 8 + sub0_size]
                    # Try to find Update Packet (Sub-chunk 1)
                    offset = 8 + sub0_size
                    if offset + 8 <= chunk_size:
                        sub1_id, sub1_size = struct.unpack('<II', chunk_data[offset: offset + 8])
                        if sub1_id == 1:
                            update_packet = chunk_data[offset + 8: offset + 8 + sub1_size]

            if not is_wrapper:
                # Legacy level.spawn format: data is just the spawn packet (no size prefix)
                spawn_packet = struct.pack('<H', len(chunk_data)) + chunk_data
                update_packet = None

            # Extract name
            name = extract_entity_name(spawn_packet)
            if name:
                entities[name] = (spawn_packet, update_packet)

    return entities


def collect_level_entities(
    level_name: str,
    level_spawn_path: Path,
    old_spawn_path: Optional[Path],
    blacklist_exact: Set[str],
    blacklist_patterns: List[str]
) -> Tuple[List[Tuple[bytes, Optional[bytes]]], int, int]:
    """
    Collect and merge entities for a level (Phase 1 extraction).
    Preserves M_UPDATE packets if they exist in the old spawn data.

    This function performs read-only extraction - no GVID remapping.
    Graph points are counted but not included in the returned entities.

    Args:
        level_name: Name of the level
        level_spawn_path: Path to the level.spawn file
        old_spawn_path: Optional path to old spawn file for merging
        blacklist_exact: Set of exact entity/section names to exclude
        blacklist_patterns: List of wildcard patterns to exclude

    Returns:
        Tuple of:
        - List of (spawn_packet, update_packet) tuples for non-graph_point entities
        - Count of graph points found
        - Count of blacklisted entities
    """
    log(f"  Collecting entities for {level_name}...")

    # 1. Load old spawn entities (Name -> (SpawnPacket, UpdatePacket))
    old_entities_by_name = {}
    if old_spawn_path:
        if isinstance(old_spawn_path, str):
            old_spawn_path = Path(old_spawn_path)
        if old_spawn_path.exists():
            old_entities_by_name = load_entities_from_spawn_file(old_spawn_path)
            log(f"    Loaded {len(old_entities_by_name)} entities from old spawn for merging")

    used_old_entities = set()
    merged_packets = []
    graph_point_count = 0
    merged_count = 0
    blacklisted_count = 0

    # 2. Process New Level Spawn
    with open(level_spawn_path, 'rb') as f:
        entity_count = 0

        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break

            chunk_id, chunk_size = struct.unpack('<II', chunk_header)
            chunk_data = f.read(chunk_size)

            if len(chunk_data) != chunk_size:
                break

            # Level.spawn usually only has M_SPAWN data without size prefix
            packet_size = len(chunk_data)
            spawn_packet = struct.pack('<H', packet_size) + chunk_data

            # New files don't have update packets
            update_packet = None

            # Extract info
            entity_name = extract_entity_name(spawn_packet)
            section_name = extract_section_name(spawn_packet)

            # Check blacklist
            if is_blacklisted(entity_name, section_name, blacklist_exact, blacklist_patterns):
                blacklisted_count += 1
                continue

            # MERGE LOGIC
            if entity_name and entity_name in old_entities_by_name:
                old_spawn, old_update = old_entities_by_name[entity_name]

                # Use old packet if it has more data (custom_data, etc)
                if len(old_spawn) > len(spawn_packet):
                    final_spawn = old_spawn
                    final_update = old_update  # Preserve existing update packet!
                    merged_count += 1
                else:
                    # Use NEW spawn data
                    final_spawn = spawn_packet
                    # Use OLD update packet if available (preserve physics state/ammo from old build)
                    final_update = old_update

                used_old_entities.add(entity_name)
            else:
                # New entity, no history
                final_spawn = spawn_packet
                final_update = None

            # Skip graph points - they are counted but not added to spawn entities
            if section_name == 'graph_point':
                graph_point_count += 1
                continue

            # Store tuple
            merged_packets.append((final_spawn, final_update))
            entity_count += 1

    # 3. Add remaining Old Entities (not in new file)
    added_old = 0
    blacklisted_old = 0
    for name, (old_spawn, old_update) in old_entities_by_name.items():
        if name not in used_old_entities:
            section = extract_section_name(old_spawn)

            # Skip graph points (they are level specific)
            if section == 'graph_point':
                continue

            if is_blacklisted(name, section, blacklist_exact, blacklist_patterns):
                blacklisted_old += 1
                blacklisted_count += 1
                continue

            merged_packets.append((old_spawn, old_update))
            added_old += 1

    log(f"    Added {entity_count} entities from level.spawn")
    if blacklisted_count - blacklisted_old > 0:
        log(f"    Blacklisted {blacklisted_count - blacklisted_old} entities from level.spawn")
    if merged_count > 0:
        log(f"    Merged {merged_count} entities with old spawn data")
    if added_old > 0:
        log(f"    Added {added_old} entities only in old spawn")
    if blacklisted_old > 0:
        log(f"    Blacklisted {blacklisted_old} entities from old spawn")

    log(f"    Graph points: {graph_point_count}")
    log(f"    Total entities for level: {len(merged_packets)}")

    return merged_packets, graph_point_count, blacklisted_count
