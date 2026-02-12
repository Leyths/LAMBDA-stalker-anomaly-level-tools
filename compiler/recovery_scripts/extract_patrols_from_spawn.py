#!/usr/bin/env python3
"""
All.Spawn Patrol Path Extractor

Extracts chunk 3 (patrol paths) from all.spawn and splits them by level.
Saves each level's patrol paths to a binary intermediate format.

This allows extracting patrol paths from an all.spawn that ACDC can't read.
"""

import struct
import io
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import sys

# Add parent directory to path for parsers import
sys.path.insert(0, str(Path(__file__).parent.parent))
from parsers import read_stringz, ChunkReader


def read_chunk(f) -> Tuple[Optional[int], Optional[bytes]]:
    """Read a chunk from file"""
    chunk_header = f.read(8)
    if len(chunk_header) < 8:
        return None, None

    chunk_id, chunk_size = struct.unpack('<II', chunk_header)
    chunk_data = f.read(chunk_size)

    if len(chunk_data) != chunk_size:
        return None, None

    return chunk_id, chunk_data


def parse_patrol_point(data: bytes, offset: int) -> Tuple[dict, int]:
    """
    Parse a CPatrolPoint from binary data

    Format (from patrol_point.cpp):
    - shared_str name (stringZ)
    - Fvector position (3 floats)
    - u32 flags
    - u32 level_vertex_id
    - u16 game_vertex_id

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


def parse_patrol_path(path_data: bytes) -> dict:
    """
    Parse a CPatrolPath from binary data

    Format (CGraphAbstractSerialize):
    - Chunk 0: vertex_count (u32)
    - Chunk 1: Vertices
    - Chunk 2: Edges

    Returns:
        patrol_dict with 'points' and 'edges'
    """
    patrol = {
        'points': [],
        'edges': {}
    }

    offset = 0

    # Read chunks
    while offset < len(path_data):
        if offset + 8 > len(path_data):
            break

        chunk_id = struct.unpack_from('<I', path_data, offset)[0]
        chunk_size = struct.unpack_from('<I', path_data, offset + 4)[0]
        offset += 8

        if offset + chunk_size > len(path_data):
            break

        chunk_data = path_data[offset:offset + chunk_size]
        offset += chunk_size

        if chunk_id == 0:
            # Vertex count
            vertex_count = struct.unpack_from('<I', chunk_data, 0)[0]
            patrol['vertex_count'] = vertex_count

        elif chunk_id == 1:
            # Vertices - parse nested chunks
            v_offset = 0
            while v_offset < len(chunk_data):
                if v_offset + 8 > len(chunk_data):
                    break

                v_chunk_id = struct.unpack_from('<I', chunk_data, v_offset)[0]
                v_chunk_size = struct.unpack_from('<I', chunk_data, v_offset + 4)[0]
                v_offset += 8

                if v_offset + v_chunk_size > len(chunk_data):
                    break

                v_chunk_data = chunk_data[v_offset:v_offset + v_chunk_size]
                v_offset += v_chunk_size

                # This is a vertex - parse its sub-chunks
                vertex_data = parse_vertex_subchunks(v_chunk_data)
                if vertex_data:
                    patrol['points'].append(vertex_data)

        elif chunk_id == 2:
            # Edges
            e_offset = 0
            while e_offset + 4 <= len(chunk_data):
                vertex_id = struct.unpack_from('<I', chunk_data, e_offset)[0]
                e_offset += 4

                if e_offset + 4 > len(chunk_data):
                    break

                edge_count = struct.unpack_from('<I', chunk_data, e_offset)[0]
                e_offset += 4

                edges = []
                for _ in range(edge_count):
                    if e_offset + 8 > len(chunk_data):
                        break

                    target_id = struct.unpack_from('<I', chunk_data, e_offset)[0]
                    weight = struct.unpack_from('<f', chunk_data, e_offset + 4)[0]
                    e_offset += 8

                    edges.append({'target': target_id, 'weight': weight})

                patrol['edges'][vertex_id] = edges

    return patrol


def parse_vertex_subchunks(vertex_data: bytes) -> dict:
    """Parse vertex sub-chunks (ID and CPatrolPoint)"""
    vertex = {}
    offset = 0

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

        if chunk_id == 0:
            # Vertex ID
            vertex['id'] = struct.unpack_from('<I', chunk_data, 0)[0]

        elif chunk_id == 1:
            # CPatrolPoint data
            point, _ = parse_patrol_point(chunk_data, 0)
            if point:
                vertex.update(point)

    return vertex


def extract_patrol_paths_from_spawn(spawn_path: Path) -> Dict[str, bytes]:
    """
    Extract all patrol paths from all.spawn chunk 3

    Returns:
        Dict of {patrol_name: patrol_binary_data}
    """
    print(f"Extracting patrol paths from {spawn_path}...")

    with open(spawn_path, 'rb') as f:
        # Find chunk 3 (patrol paths)
        while True:
            chunk_id, chunk_data = read_chunk(f)

            if chunk_id is None:
                print("  Error: Could not find chunk 3 (patrol paths)")
                return {}

            if chunk_id == 3:
                print(f"  Found chunk 3: {len(chunk_data):,} bytes")
                break

    # Parse chunk 3 (patrol path storage)
    patrols = {}
    offset = 0

    # Chunk 3 format:
    # - Chunk 0: patrol_count (u32)
    # - Chunk 1: Patrol paths

    while offset < len(chunk_data):
        if offset + 8 > len(chunk_data):
            break

        chunk_id = struct.unpack_from('<I', chunk_data, offset)[0]
        chunk_size = struct.unpack_from('<I', chunk_data, offset + 4)[0]
        offset += 8

        if offset + chunk_size > len(chunk_data):
            break

        inner_chunk_data = chunk_data[offset:offset + chunk_size]
        offset += chunk_size

        if chunk_id == 0:
            # Patrol count
            patrol_count = struct.unpack_from('<I', inner_chunk_data, 0)[0]
            print(f"  Patrol count: {patrol_count}")

        elif chunk_id == 1:
            # Patrol paths
            patrols = parse_patrol_paths_chunk(inner_chunk_data)
            print(f"  Extracted {len(patrols)} patrol paths")

    return patrols


def parse_patrol_paths_chunk(paths_data: bytes) -> Dict[str, bytes]:
    """Parse chunk 1 (patrol paths) and return dict of {name: binary_data}"""
    patrols = {}
    offset = 0

    while offset < len(paths_data):
        if offset + 8 > len(paths_data):
            break

        # Each patrol is a chunk
        patrol_chunk_id = struct.unpack_from('<I', paths_data, offset)[0]
        patrol_chunk_size = struct.unpack_from('<I', paths_data, offset + 4)[0]
        offset += 8

        if offset + patrol_chunk_size > len(paths_data):
            break

        patrol_data = paths_data[offset:offset + patrol_chunk_size]
        offset += patrol_chunk_size

        # Parse patrol name and data
        name, path_binary = parse_single_patrol(patrol_data)
        if name:
            patrols[name] = path_binary

    return patrols


def parse_single_patrol(patrol_data: bytes) -> Tuple[str, bytes]:
    """
    Parse a single patrol path chunk

    Format:
    - Sub-chunk 0: name (stringZ)
    - Sub-chunk 1: CPatrolPath data
    """
    name = ""
    path_data = b""

    offset = 0
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

        if chunk_id == 0:
            # Name
            name, _ = read_stringz(chunk_data, 0)

        elif chunk_id == 1:
            # CPatrolPath data (keep as binary)
            path_data = chunk_data

    return name, path_data


def map_patrol_to_level(patrol_name: str, patrol_data: bytes,
                        game_graph_data: bytes) -> str:
    """
    Determine which level a patrol path belongs to based on its game_vertex_ids

    Args:
        patrol_name: Name of the patrol
        patrol_data: Binary patrol path data
        game_graph_data: Binary game graph data

    Returns:
        Level name, or 'unknown' if can't determine
    """
    # Parse patrol to get game_vertex_ids
    patrol = parse_patrol_path(patrol_data)

    if not patrol['points']:
        return 'unknown'

    # Map level using GVID from patrol points
    for point in patrol['points']:
        gvid = point.get('game_vertex_id', 0xFFFF)
        if gvid != 0xFFFF and gvid != 0:
            level_name = lookup_level_from_game_graph(gvid, game_graph_data)
            if level_name:
                return level_name

    # GVID lookup failed (corrupt data) - fall back to name prefix
    return guess_level_from_patrol_name(patrol_name)


def guess_level_from_patrol_name(patrol_name: str) -> str:
    """
    Guess level from patrol path name prefix.

    Only used as a fallback for patrols with corrupt GVIDs.
    """
    # Explicit overrides for patrols where the prefix is ambiguous
    # or too generic to add as a prefix rule
    name_overrides = {
        'pri_space_restrictor_to_cop_pripyat_drop': 'l11_pripyat',
        # l11_hospital zone combat routes (z1/z2/z3 hospital zones)
        'actors_way': 'l05_bar',
        'barricade_final_3_look': 'l11_hospital',
        'barricade_fire_2_look': 'l11_hospital',
        'barricade_fire_3_look': 'l11_hospital',
        'barricade_fire_4_look': 'l11_hospital',
        'enemy_floor2l_look': 'l11_hospital',
        'heli_death_way': 'l12_stancia',
        'heli_z2_ne_walk': 'l11_hospital',
        'heli_z2_se_walk': 'l11_hospital',
        'minigun_fire_way_1': 'l11_hospital',
        'minigun_fire_way_5': 'l11_hospital',
        'minigun_fire_way_bridge_1': 'l11_hospital',
        'minigun_fire_way_bridge_2': 'l11_hospital',
        'minigun_fire_way_bridge_5': 'l11_hospital',
        'sniper_z1_1_look_hide': 'l11_hospital',
        'sniper_z1_3_look': 'l11_hospital',
        'teleport_way2': 'l10_limansk',
        'z1_minigunner_look': 'l11_hospital',
        'z2_heli_hide': 'l11_hospital',
        'z2_heli_move': 'l11_hospital',
        'z2_heli_move_crash': 'l11_hospital',
        'z2_heli_move_crash_0': 'l11_hospital',
        'z2_heli_move_crash_look': 'l11_hospital',
        'z2_heli_move_nw': 'l11_hospital',
        'z2_heli_move_s': 'l11_hospital',
        'z2_heli_move_sw_nw': 'l11_hospital',
        'z2_heli_show_fire_2': 'l11_hospital',
        'z2_heli_show_fire_3': 'l11_hospital',
        'z2_heli_show_fire_6': 'l11_hospital',
        'z2_particle_1': 'l11_hospital',
        'z2_particle_3': 'l11_hospital',
        'z3_enemy_floor2_1_look_0': 'l11_hospital',
        'z3_jump_way_1': 'l11_hospital',
        'z3_minigun_fire_1': 'l11_hospital',
        'z3_minigun_fire_3': 'l11_hospital',
    }
    if patrol_name in name_overrides:
        return name_overrides[patrol_name]

    # Common prefixes - mapped to current level names from levels.ini
    # Longer prefixes must come first to match correctly (e.g. 'aes2_' before 'aes_')
    prefix_to_level = {
        # Short name prefixes
        'esc_': 'l01_escape',
        'gar_': 'l02_garbage',
        'agr_': 'l03_agroprom',
        'dar_': 'l04_darkvalley',
        'bar_': 'l05_bar',
        'ros_': 'l06_rostok',
        'mil_': 'l07_military',
        'yan_': 'l08_yantar',
        'jup_': 'jupiter',
        'zat_': 'zaton',
        'pri_': 'pripyat',
        'mar_': 'k00_marsh',
        'val_': 'l11_hospital',
        'red_': 'l10_red_forest',
        'lim_': 'l10_limansk',
        'pas_': 'l11_pripyat',
        'rad_': 'l10_radar',
        'aes2_': 'l12_stancia_2',
        'aes_': 'l12_stancia',
        'ds_': 'k01_darkscape',
        'cop_': 'pripyat',
        'tc_': 'k02_trucks_cemetery',
        'pol_': 'y04_pole',
        # Full level name prefixes
        'k00_': 'k00_marsh',
        'k01_': 'k01_darkscape',
        'l01_': 'l01_escape',
        'l02_': 'l02_garbage',
        'l03u_': 'l03u_agr_underground',
        'l03_': 'l03_agroprom',
        'l04u_': 'l04u_labx18',
        'l04_': 'l04_darkvalley',
        'l05_': 'l05_bar',
        'l06_': 'l06_rostok',
        'l07_': 'l07_military',
        'l08u_': 'l08u_brainlab',
        'l08_': 'l08_yantar',
        'l10u_': 'l10u_bunker',
        'l10_limansk': 'l10_limansk',
        'l10_': 'l10_radar',
        'l11_pripyat': 'l11_pripyat',
        'l11_': 'l11_hospital',
        'l12u_': 'l12u_sarcofag',
        'l12_': 'l12_stancia',
        'l13_': 'l13_generators',
        # Underground/interior prefixes
        'x16_': 'l03u_agr_underground',
        'und_': 'l03u_agr_underground',
        'sar_': 'l12u_sarcofag',
        'bun_': 'l10u_bunker',
        'cit_': 'l09_deadcity',
        'katacomb_': 'l11_hospital',
        'warlab_': 'l13u_warlab',
        'zaton_': 'zaton',
        'rostok_': 'l06_rostok',
    }

    # Check each prefix
    for prefix, level in prefix_to_level.items():
        if patrol_name.startswith(prefix):
            return level

    # Can't guess
    return 'unknown'


def lookup_level_from_game_graph(game_vertex_id: int, game_graph_data: bytes) -> str:
    """Look up level name from game vertex ID in game graph"""
    try:
        offset = 0

        # Parse game graph header
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

        # Parse levels
        levels = {}
        for i in range(level_count):
            level_name, offset = read_stringz(game_graph_data, offset)
            level_offset = struct.unpack_from('<3f', game_graph_data, offset)
            offset += 12
            level_id = struct.unpack_from('<B', game_graph_data, offset)[0]
            offset += 1
            level_section, offset = read_stringz(game_graph_data, offset)
            level_guid = game_graph_data[offset:offset + 16]
            offset += 16

            levels[level_id] = level_name

        # Parse vertices to find which level this game_vertex_id belongs to
        VERTEX_SIZE = 42
        vertices_start = offset

        if game_vertex_id >= vertex_count:
            return None

        v_offset = vertices_start + game_vertex_id * VERTEX_SIZE

        # Read vertex
        # Skip local and global points
        v_offset += 24

        # Read packed level_id and node_id
        packed = struct.unpack_from('<I', game_graph_data, v_offset)[0]
        level_id = packed & 0xFF

        return levels.get(level_id)

    except Exception as e:
        print(f"  Warning: Could not lookup level for gvid {game_vertex_id}: {e}")
        return None


def extract_game_graph_from_spawn(spawn_path: Path) -> bytes:
    """
    Extract game graph (chunk 4) from all.spawn

    Args:
        spawn_path: Path to all.spawn

    Returns:
        Game graph binary data
    """
    print(f"Extracting game graph from {spawn_path}...")

    with open(spawn_path, 'rb') as f:
        # Find chunk 4 (game graph)
        while True:
            chunk_id, chunk_data = read_chunk(f)

            if chunk_id is None:
                print("  Error: Could not find chunk 4 (game graph)")
                return b""

            if chunk_id == 4:
                print(f"  Found chunk 4 (game graph): {len(chunk_data):,} bytes")
                return chunk_data


def extract_and_split_patrols(spawn_path: Path, output_dir: Path):
    """
    Extract patrol paths from all.spawn and split by level

    Args:
        spawn_path: Path to all.spawn
        output_dir: Output directory for level-specific patrol files
    """
    # Extract game graph from the same all.spawn file
    game_graph_data = extract_game_graph_from_spawn(spawn_path)

    if not game_graph_data:
        print("ERROR: Could not extract game graph from all.spawn!")
        print("The game graph is needed to determine which level each patrol belongs to.")
        return

    # Extract all patrol paths
    patrols = extract_patrol_paths_from_spawn(spawn_path)

    if not patrols:
        print("No patrol paths found!")
        return

    import re

    # Pass 1: Map patrols by GVID
    gvid_mapped = {}   # patrol_name -> level_name
    gvid_unmapped = {} # patrol_name -> patrol_data (GVID lookup failed)

    for patrol_name, patrol_data in patrols.items():
        patrol = parse_patrol_path(patrol_data)
        level_name = None

        if patrol['points']:
            for point in patrol['points']:
                gvid = point.get('game_vertex_id', 0xFFFF)
                if gvid != 0xFFFF and gvid != 0:
                    level_name = lookup_level_from_game_graph(gvid, game_graph_data)
                    if level_name:
                        break

        if level_name:
            gvid_mapped[patrol_name] = level_name
        else:
            gvid_unmapped[patrol_name] = patrol_data

    # Pass 2: Build smart terrain prefix -> level lookup from GVID-mapped patrols
    # e.g. "esc_smart_terrain_5_7" -> "l01_escape"
    smart_terrain_re = re.compile(r'^(.+_smart_terrain_\d+_\d+)')
    smart_terrain_levels = {}

    for patrol_name, level_name in gvid_mapped.items():
        m = smart_terrain_re.match(patrol_name)
        if m:
            smart_terrain_levels[m.group(1)] = level_name

    # Pass 3: Try to resolve unmapped patrols via smart terrain prefix match,
    # then fall back to name prefix table
    smart_terrain_resolved = 0
    prefix_resolved = 0
    still_unknown = 0

    for patrol_name in list(gvid_unmapped.keys()):
        # Check explicit overrides first
        level_name = None

        # Try smart terrain prefix match
        m = smart_terrain_re.match(patrol_name)
        if m and m.group(1) in smart_terrain_levels:
            level_name = smart_terrain_levels[m.group(1)]
            smart_terrain_resolved += 1

        # Fall back to name prefix table
        if not level_name:
            level_name = guess_level_from_patrol_name(patrol_name)
            if level_name != 'unknown':
                prefix_resolved += 1
            else:
                still_unknown += 1

        gvid_mapped[patrol_name] = level_name
        del gvid_unmapped[patrol_name]

    # Group by level
    patrols_by_level = {}
    for patrol_name, level_name in gvid_mapped.items():
        if level_name not in patrols_by_level:
            patrols_by_level[level_name] = {}
        patrols_by_level[level_name][patrol_name] = patrols[patrol_name]

    # Write per-level files
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting level-specific patrol files:")

    total_count = len(patrols)
    gvid_count = total_count - smart_terrain_resolved - prefix_resolved - still_unknown

    print(f"\n  Mapped by GVID: {gvid_count}")
    print(f"  Mapped by smart terrain match: {smart_terrain_resolved}")
    if prefix_resolved > 0:
        print(f"  Mapped by name prefix: {prefix_resolved}")
    if still_unknown > 0:
        print(f"  Unmapped: {still_unknown}")

    for level_name, level_patrols in sorted(patrols_by_level.items()):
        output_file = output_dir / f"{level_name}.patrols"

        # Write as simple binary format:
        # - u32 patrol_count
        # - For each patrol:
        #   - u16 name_length
        #   - name (bytes)
        #   - u32 data_length
        #   - data (bytes)

        with open(output_file, 'wb') as f:
            f.write(struct.pack('<I', len(level_patrols)))

            for patrol_name, patrol_data in sorted(level_patrols.items()):
                name_bytes = patrol_name.encode('utf-8')
                f.write(struct.pack('<H', len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack('<I', len(patrol_data)))
                f.write(patrol_data)

        print(f"  {level_name}: {len(level_patrols)} patrols -> {output_file}")

    print(f"\nExtraction complete! Files saved to {output_dir}/")
    print(f"Total patrols extracted: {len(patrols)}")
    print(f"Levels with patrols: {len(patrols_by_level)}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python extract_patrols_from_spawn.py <all.spawn> <output_dir>")
        print("\nExtracts patrol paths from all.spawn and splits them by level.")
        print("Saves each level's patrols to <level_name>.patrols")
        print("\nArguments:")
        print("  <all.spawn>   - Original Anomaly all.spawn file to extract from")
        print("                  (from Anomaly's gamedata/spawns/all.spawn)")
        print("                  The game graph is extracted from chunk 4 of this file.")
        print("  <output_dir>  - Directory to save extracted .patrols files")
        print("\nNote: Use the ORIGINAL Anomaly all.spawn, not your generated one!")
        print("      Your build process will later merge these with new patrols.")
        sys.exit(1)

    spawn_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not spawn_path.exists():
        print(f"Error: {spawn_path} not found")
        sys.exit(1)

    extract_and_split_patrols(spawn_path, output_dir)