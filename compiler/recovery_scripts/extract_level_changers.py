#!/usr/bin/env python3
"""
Extract Level Changers from all.spawn

Parses an all.spawn file and generates a level_changers.ini config file
containing all level changer destinations, organized by source level.

Usage:
    python extract_level_changers.py <all.spawn> <output.ini>

Example:
    python extract_level_changers.py ../../misc/all.spawn ../../level_changers.ini
"""

import struct
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add parent directory to path for parsers import
sys.path.insert(0, str(Path(__file__).parent.parent))
from parsers import (
    read_stringz,
    ChunkReader,
    parse_level_changer_packet,
)


@dataclass
class LevelInfo:
    """Level information from game graph header."""
    level_id: int
    name: str


@dataclass
class LevelChangerEntry:
    """Extracted level changer data."""
    entity_name: str
    source_level: str
    dest_level_name: str
    dest_position: Tuple[float, float, float]    # (x, y, z)
    dest_direction: Tuple[float, float, float]   # (dir_x, dir_y, dir_z)
    dest_gvid: int
    source_gvid: int


def parse_game_graph_levels(game_graph_data: bytes) -> Tuple[Dict[int, LevelInfo], Dict[int, int]]:
    """
    Parse game graph header to extract level definitions and vertex-to-level mapping.

    Returns:
        Tuple of (level_id -> LevelInfo, game_vertex_id -> level_id)
    """
    levels = {}
    vertex_to_level = {}

    offset = 0

    # Header format:
    #   u8 m_version
    #   u16 m_vertex_count
    #   u32 m_edge_count
    #   u32 m_death_point_count
    #   xrGUID m_guid (16 bytes)
    #   u8 level_count

    version = struct.unpack_from('<B', game_graph_data, offset)[0]
    offset += 1

    vertex_count = struct.unpack_from('<H', game_graph_data, offset)[0]
    offset += 2

    edge_count = struct.unpack_from('<I', game_graph_data, offset)[0]
    offset += 4

    death_point_count = struct.unpack_from('<I', game_graph_data, offset)[0]
    offset += 4

    guid = game_graph_data[offset:offset + 16]
    offset += 16

    level_count = struct.unpack_from('<B', game_graph_data, offset)[0]
    offset += 1

    print(f"  Game graph version: {version}")
    print(f"  Vertex count: {vertex_count}")
    print(f"  Level count: {level_count}")

    # Parse level definitions
    for i in range(level_count):
        level_name, offset = read_stringz(game_graph_data, offset)
        level_offset = struct.unpack_from('<3f', game_graph_data, offset)
        offset += 12
        level_id = struct.unpack_from('<B', game_graph_data, offset)[0]
        offset += 1
        level_section, offset = read_stringz(game_graph_data, offset)
        level_guid = game_graph_data[offset:offset + 16]
        offset += 16

        levels[level_id] = LevelInfo(level_id=level_id, name=level_name)

    # Parse vertices to build vertex_to_level mapping
    VERTEX_SIZE = 42
    vertices_start = offset

    for vertex_id in range(vertex_count):
        v_offset = vertices_start + vertex_id * VERTEX_SIZE

        # Skip local_point (12) and global_point (12)
        # Read packed level_id:8, node_id:24
        packed = struct.unpack_from('<I', game_graph_data, v_offset + 24)[0]
        level_id = packed & 0xFF

        vertex_to_level[vertex_id] = level_id

    return levels, vertex_to_level


def extract_level_changers(spawn_graph_data: bytes, levels: Dict[int, LevelInfo],
                           vertex_to_level: Dict[int, int]) -> List[LevelChangerEntry]:
    """
    Extract level changers from spawn graph chunk.

    Args:
        spawn_graph_data: Spawn graph chunk data (chunk 1)
        levels: Level definitions
        vertex_to_level: Vertex to level mapping

    Returns:
        List of LevelChangerEntry objects
    """
    level_changers = []

    # Parse spawn graph structure
    sg_offset = 0
    vertices_data = None

    while sg_offset < len(spawn_graph_data) - 8:
        sub_id, sub_size = struct.unpack_from('<II', spawn_graph_data, sg_offset)
        sub_data = spawn_graph_data[sg_offset + 8:sg_offset + 8 + sub_size]

        if sub_id == 1:  # Vertices chunk
            vertices_data = sub_data

        sg_offset += 8 + sub_size

    if not vertices_data:
        print("  ERROR: No vertices data found in spawn graph!")
        return level_changers

    # Parse each spawn vertex
    v_offset = 0

    while v_offset < len(vertices_data) - 8:
        v_chunk_id, v_chunk_size = struct.unpack_from('<II', vertices_data, v_offset)
        v_chunk_data = vertices_data[v_offset + 8:v_offset + 8 + v_chunk_size]

        # Parse vertex sub-chunks
        vv_offset = 0
        vertex_id = 0
        spawn_packet = None

        while vv_offset < len(v_chunk_data) - 8:
            vv_id, vv_size = struct.unpack_from('<II', v_chunk_data, vv_offset)
            vv_data = v_chunk_data[vv_offset + 8:vv_offset + 8 + vv_size]

            if vv_id == 0:
                vertex_id = struct.unpack_from('<H', vv_data, 0)[0]
            elif vv_id == 1:
                # CServerEntityWrapper - extract M_SPAWN packet
                wrapper_offset = 0
                while wrapper_offset < len(vv_data) - 8:
                    w_id, w_size = struct.unpack_from('<II', vv_data, wrapper_offset)
                    w_data = vv_data[wrapper_offset + 8:wrapper_offset + 8 + w_size]

                    if w_id == 0:  # M_SPAWN packet
                        spawn_packet = w_data

                    wrapper_offset += 8 + w_size

            vv_offset += 8 + vv_size

        # Process spawn packet if it's a level_changer
        if spawn_packet:
            entry = process_spawn_packet(spawn_packet, levels, vertex_to_level)
            if entry:
                level_changers.append(entry)

        v_offset += 8 + v_chunk_size

    return level_changers


def process_spawn_packet(spawn_packet: bytes, levels: Dict[int, LevelInfo],
                         vertex_to_level: Dict[int, int]) -> Optional[LevelChangerEntry]:
    """
    Process a spawn packet and extract level changer data if applicable.

    Args:
        spawn_packet: M_SPAWN packet data
        levels: Level definitions
        vertex_to_level: Vertex to level mapping

    Returns:
        LevelChangerEntry if this is a level_changer, None otherwise
    """
    try:
        # Quick check for level_changer section
        offset = 0

        # Size prefix check
        if len(spawn_packet) >= 4:
            potential_size = struct.unpack_from('<H', spawn_packet, 0)[0]
            if potential_size == len(spawn_packet) - 2:
                offset = 2

        # M_SPAWN type
        offset += 2

        # Section name
        section_name, offset = read_stringz(spawn_packet, offset)
        if section_name != 'level_changer':
            return None

        # Entity name
        entity_name, offset = read_stringz(spawn_packet, offset)

        # Skip to game_vertex_id
        # gameid (1) + rp (1) + position (12) + angle (12) + respawn/id/parent/phantom (8)
        offset += 34

        # s_flags
        s_flags = struct.unpack_from('<H', spawn_packet, offset)[0]
        offset += 2

        # version
        version = 0
        if s_flags & 0x20:
            version = struct.unpack_from('<H', spawn_packet, offset)[0]
            offset += 2

        if version > 120:
            offset += 2  # game_type
        if version > 69:
            offset += 2  # script_version
        if version > 70:
            client_data_size = struct.unpack_from('<H', spawn_packet, offset)[0]
            offset += 2 + client_data_size
        if version > 79:
            offset += 2  # spawn_id

        # data_size
        offset += 2

        # game_vertex_id
        source_gvid = struct.unpack_from('<H', spawn_packet, offset)[0]

        # Parse full level_changer data using the parser
        lc_data = parse_level_changer_packet(spawn_packet)
        if lc_data is None:
            return None

        # Determine source level from game_vertex_id
        source_level_id = vertex_to_level.get(source_gvid, -1)
        source_level = levels.get(source_level_id, LevelInfo(-1, "unknown")).name

        return LevelChangerEntry(
            entity_name=entity_name,
            source_level=source_level,
            dest_level_name=lc_data.dest_level_name,
            dest_position=lc_data.dest_position,
            dest_direction=lc_data.dest_direction,
            dest_gvid=lc_data.dest_game_vertex_id,
            source_gvid=source_gvid
        )

    except Exception as e:
        return None


def write_level_changers_ini(level_changers: List[LevelChangerEntry], output_path: Path):
    """
    Write level changers to INI format.

    Args:
        level_changers: List of extracted level changers
        output_path: Output INI file path
    """
    # Group by source level
    by_level = defaultdict(list)
    for lc in level_changers:
        by_level[lc.source_level].append(lc)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Level Changers Configuration\n")
        f.write("# Each level changer has 3 keys: dest, pos, dir\n")
        f.write("#\n")
        f.write("# Format:\n")
        f.write("#   entity_name.dest = destination_level\n")
        f.write("#   entity_name.pos = x, y, z          (world coordinates)\n")
        f.write("#   entity_name.dir = pitch, yaw, roll (radians)\n")
        f.write("#\n")
        f.write("# Direction (dir) is the actor's camera orientation on arrival:\n")
        f.write("#   - pitch: vertical angle (positive = looking up, negative = looking down)\n")
        f.write("#   - yaw: horizontal angle (rotation around vertical axis)\n")
        f.write("#   - roll: tilt angle (usually 0)\n")
        f.write("#\n")
        f.write("# To convert radians to degrees: degrees = radians * 180 / pi\n")
        f.write("#   Examples: 1.57 rad = 90°, 3.14 rad = 180°, -1.57 rad = -90°\n")
        f.write("#\n")
        f.write("# Config is authoritative - level changers NOT in this file are removed from all.spawn\n")
        f.write("#\n\n")

        for source_level in sorted(by_level.keys()):
            changers = by_level[source_level]
            f.write(f"[{source_level}]\n")

            for lc in sorted(changers, key=lambda x: x.entity_name):
                # Format: entity_name.dest, entity_name.pos, entity_name.dir
                x, y, z = lc.dest_position
                dx, dy, dz = lc.dest_direction
                f.write(f"{lc.entity_name}.dest = {lc.dest_level_name}\n")
                f.write(f"{lc.entity_name}.pos = {x:.2f}, {y:.2f}, {z:.2f}\n")
                f.write(f"{lc.entity_name}.dir = {dx:.2f}, {dy:.2f}, {dz:.2f}\n\n")

            f.write("\n")

    print(f"\nWrote {len(level_changers)} level changers to {output_path}")
    print(f"  Levels with changers: {len(by_level)}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_level_changers.py <all.spawn> <output.ini>")
        print("\nExtracts level changers from all.spawn and generates level_changers.ini")
        sys.exit(1)

    spawn_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not spawn_path.exists():
        print(f"ERROR: File not found: {spawn_path}")
        sys.exit(1)

    print(f"Parsing: {spawn_path}")

    with open(spawn_path, 'rb') as f:
        data = f.read()

    print(f"  File size: {len(data):,} bytes")

    # Find chunks
    offset = 0
    game_graph_data = None
    spawn_graph_data = None

    while offset < len(data) - 8:
        chunk_id, chunk_size = struct.unpack_from('<II', data, offset)
        chunk_data = data[offset + 8:offset + 8 + chunk_size]

        if chunk_id == 1:
            spawn_graph_data = chunk_data
            print(f"  Found spawn graph chunk (1): {chunk_size:,} bytes")
        elif chunk_id == 4:
            game_graph_data = chunk_data
            print(f"  Found game graph chunk (4): {chunk_size:,} bytes")

        offset += 8 + chunk_size

    if not game_graph_data:
        print("ERROR: No game graph chunk found!")
        sys.exit(1)

    if not spawn_graph_data:
        print("ERROR: No spawn graph chunk found!")
        sys.exit(1)

    # Parse game graph
    levels, vertex_to_level = parse_game_graph_levels(game_graph_data)

    # Extract level changers
    print("\nExtracting level changers...")
    level_changers = extract_level_changers(spawn_graph_data, levels, vertex_to_level)

    print(f"  Found {len(level_changers)} level changers")

    # Print summary by level
    by_level = defaultdict(list)
    for lc in level_changers:
        by_level[lc.source_level].append(lc)

    print("\n  Level changers by source level:")
    for source_level in sorted(by_level.keys()):
        changers = by_level[source_level]
        print(f"    {source_level}: {len(changers)}")

    # Write INI file
    write_level_changers_ini(level_changers, output_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
