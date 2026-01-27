#!/usr/bin/env python3
"""
Spawn Extractor V4 - With Graph Point Extraction

Extracts entities from all.spawn and splits them by level based on
the GAME GRAPH structure, not entity name prefixes.

Also extracts graph points from the GAME GRAPH VERTICES since graph_point
entities are compiled into game graph vertices during the build process and
are NOT stored in the spawn entities section.

Supports two modes:

1. Single-source mode (extract ALL entities):
   python extract_spawns.py <all.spawn> <output_dir>

2. Comparison mode (extract only differences):
   python extract_spawns.py <all.spawn> <output_dir> --original <original_all.spawn>

Flow:
1. Parse game graph header to get level definitions (level_id -> level_name)
2. Parse game graph vertices to:
   - Build game_vertex_id -> level_id mapping
   - Extract graph point data (position, level_vertex_id, location types)
3. Parse spawn entities and assign to levels based on their game_vertex_id
4. (Comparison mode only) Compare with original spawn and extract entities that differ
5. Create graph_point spawn packets from game graph vertices
6. Write per-level spawn files containing both entities AND graph points
"""

import struct
import sys
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import json

# Add parent directory to path for parsers import
sys.path.insert(0, str(Path(__file__).parent.parent))
from parsers import read_stringz, GameGraphParser, ChunkReader


def parse_graph_point_names_ini(ini_path: Path) -> Dict[str, Dict[Tuple[float, float, float], str]]:
    """
    Parse INI file mapping coordinates to custom graph point names.

    INI format:
        [level_name]
        point_name,x,y,z
        # comments allowed

    Returns: Dict[level_name, Dict[(x,y,z), point_name]]
    """
    result = {}
    current_level = None

    with open(ini_path, 'r') as f:
        for line in f:
            line = line.split('#')[0].strip()  # Remove comments
            if not line:
                continue

            if line.startswith('[') and line.endswith(']'):
                current_level = line[1:-1]
                result[current_level] = {}
            elif current_level and ',' in line:
                parts = line.split(',')
                if len(parts) >= 4:
                    name = parts[0].strip()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    result[current_level][(x, y, z)] = name

    return result


def find_matching_name(position: Tuple[float, float, float],
                       coord_to_name: Dict[Tuple[float, float, float], str],
                       tolerance: float = 1) -> Optional[str]:
    """Find custom name for position within tolerance (Euclidean distance)."""
    px, py, pz = position
    for (cx, cy, cz), name in coord_to_name.items():
        dist = ((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2) ** 0.5
        if dist <= tolerance:
            return name
    return None


@dataclass
class LevelInfo:
    """Level information from game graph header"""
    level_id: int
    name: str
    section: str
    offset: Tuple[float, float, float]
    guid: bytes


@dataclass
class GameGraphVertex:
    """Game graph vertex - represents a compiled graph_point"""
    vertex_id: int
    level_id: int
    level_vertex_id: int
    local_position: Tuple[float, float, float]
    global_position: Tuple[float, float, float]
    vertex_types: bytes  # 4 bytes - location types


@dataclass
class SpawnEntity:
    """Parsed spawn entity"""
    spawn_id: int
    vertex_index: int
    section_name: str
    entity_name: str
    position: Tuple[float, float, float]
    game_vertex_id: int
    level_vertex_id: int
    raw_packet: bytes  # Full spawn packet including size prefix
    update_packet: Optional[bytes] = None  # M_UPDATE packet if present

    # Parsed for comparison (excludes graph IDs)
    comparison_data: bytes = field(default=b'')


def make_entity_key(entity: SpawnEntity) -> tuple:
    """Create unique key for entity using name and rounded position.

    Entities with the same name can exist on different levels (different positions).
    Using (name, position) as key prevents deduplication of legitimate duplicates.
    Position is rounded to 2 decimal places for float comparison stability.
    """
    pos = (round(entity.position[0], 2),
           round(entity.position[1], 2),
           round(entity.position[2], 2))
    return (entity.entity_name, pos)


def parse_game_graph(game_graph_data: bytes) -> Tuple[Dict[int, LevelInfo], Dict[int, int], List[GameGraphVertex]]:
    """
    Parse game graph to extract:
    1. Level definitions (level_id -> LevelInfo)
    2. Vertex to level mapping (game_vertex_id -> level_id)
    3. Full vertex data for graph point reconstruction

    Returns: (levels_dict, vertex_to_level_dict, vertices_list)
    """
    levels = {}
    vertex_to_level = {}
    vertices = []

    offset = 0

    # Header format from CHeader::load() and game_graph_space.h:
    #   u8 m_version
    #   u16 m_vertex_count (_GRAPH_ID is u16)
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
    print(f"  Vertex count: {vertex_count} (these are the graph points!)")
    print(f"  Edge count: {edge_count}")
    print(f"  Death point count: {death_point_count}")
    print(f"  Level count: {level_count}")

    # Parse levels - SLevel::load() format:
    #   r_stringZ(m_name) - null-terminated string
    #   r_fvector3(m_offset) - 12 bytes
    #   u8 m_id
    #   r_stringZ(m_section) - null-terminated string
    #   xrGUID m_guid - 16 bytes

    for i in range(level_count):
        level_name, offset = read_stringz(game_graph_data, offset)
        level_offset = struct.unpack_from('<3f', game_graph_data, offset)
        offset += 12
        level_id = struct.unpack_from('<B', game_graph_data, offset)[0]
        offset += 1
        level_section, offset = read_stringz(game_graph_data, offset)
        level_guid = game_graph_data[offset:offset + 16]
        offset += 16

        levels[level_id] = LevelInfo(
            level_id=level_id,
            name=level_name,
            section=level_section,
            offset=level_offset,
            guid=level_guid
        )

        print(f"    Level {level_id}: {level_name} ({level_section})")

    # Parse vertices - CVertex structure (42 bytes each, from game_graph_space.h):
    #   Fvector tLocalPoint (12 bytes)
    #   Fvector tGlobalPoint (12 bytes)
    #   u32 tLevelID:8, tNodeID:24 (4 bytes packed)
    #   u8 tVertexTypes[4] (4 bytes)
    #   u32 dwEdgeOffset (4 bytes)
    #   u32 dwPointOffset (4 bytes)
    #   u8 tNeighbourCount (1 byte)
    #   u8 tDeathPointCount (1 byte)

    VERTEX_SIZE = 42
    vertices_start = offset

    print(f"  Vertices data starts at offset {vertices_start}")

    for vertex_id in range(vertex_count):
        v_offset = vertices_start + vertex_id * VERTEX_SIZE

        # Fvector tLocalPoint (12 bytes)
        local_point = struct.unpack_from('<3f', game_graph_data, v_offset)

        # Fvector tGlobalPoint (12 bytes)
        global_point = struct.unpack_from('<3f', game_graph_data, v_offset + 12)

        # u32 tLevelID:8, tNodeID:24 (4 bytes packed)
        packed = struct.unpack_from('<I', game_graph_data, v_offset + 24)[0]
        level_id = packed & 0xFF
        level_vertex_id = (packed >> 8) & 0xFFFFFF

        # u8 tVertexTypes[4] (4 bytes)
        vertex_types = game_graph_data[v_offset + 28:v_offset + 32]

        vertex_to_level[vertex_id] = level_id

        vertices.append(GameGraphVertex(
            vertex_id=vertex_id,
            level_id=level_id,
            level_vertex_id=level_vertex_id,
            local_position=local_point,
            global_position=global_point,
            vertex_types=vertex_types
        ))

    print(f"  Parsed {len(vertices)} graph point vertices")

    # Print level vertex ranges
    level_vertex_ranges = {}
    for v in vertices:
        lid = v.level_id
        vid = v.vertex_id
        if lid not in level_vertex_ranges:
            level_vertex_ranges[lid] = [vid, vid, 1]
        else:
            level_vertex_ranges[lid][0] = min(level_vertex_ranges[lid][0], vid)
            level_vertex_ranges[lid][1] = max(level_vertex_ranges[lid][1], vid)
            level_vertex_ranges[lid][2] += 1

    print(f"  Graph points by level:")
    for lid in sorted(level_vertex_ranges.keys()):
        vmin, vmax, count = level_vertex_ranges[lid]
        level_name = levels.get(lid, LevelInfo(lid, "unknown", "", (0, 0, 0), b"")).name
        print(f"    {level_name} (id={lid}): {count} graph points (vertices {vmin}-{vmax})")

    return levels, vertex_to_level, vertices


def create_graph_point_packet(vertex: GameGraphVertex, name: str) -> bytes:
    """
    Create an M_SPAWN packet for a graph_point entity from a game graph vertex.

    Format based on CSE_Abstract::Spawn_Write and CSE_ALifeGraphPoint::STATE_Write.
    Returns packet WITHOUT size prefix (for level.spawn format).
    """
    buffer = io.BytesIO()

    # M_SPAWN message type
    buffer.write(struct.pack('<H', 1))  # M_SPAWN = 1

    # Section name (stringZ)
    buffer.write(b'graph_point\x00')

    # Entity name (stringZ)
    buffer.write(name.encode('utf-8') + b'\x00')

    # gameid (u8), s_RP (u8) - 0xFE = use supplied coords
    buffer.write(struct.pack('<BB', 0, 0xFE))

    # Position (Fvector) - use local point
    buffer.write(struct.pack('<3f', *vertex.local_position))

    # Angle (Fvector) - zeros
    buffer.write(struct.pack('<3f', 0.0, 0.0, 0.0))

    # RespawnTime, ID, ID_Parent, ID_Phantom (all u16)
    buffer.write(struct.pack('<HHHH', 0, 0xFFFF, 0xFFFF, 0xFFFF))

    # s_flags (u16) - M_SPAWN_VERSION = 0x20
    buffer.write(struct.pack('<H', 0x20))

    # Version (u16) - SPAWN_VERSION = 128
    buffer.write(struct.pack('<H', 128))

    # game_type (u16) - since version > 120
    buffer.write(struct.pack('<H', 1))  # GAME_SINGLE

    # script_version (u16) - since version > 69
    buffer.write(struct.pack('<H', 8))

    # client_data_size (u16) - since version > 70
    buffer.write(struct.pack('<H', 0))

    # spawn_id (u16) - since version > 79
    buffer.write(struct.pack('<H', 0))

    # data_size placeholder
    data_start = buffer.tell()
    buffer.write(struct.pack('<H', 0))

    # CSE_ALifeObject fields:
    buffer.write(struct.pack('<H', vertex.vertex_id))  # game_vertex_id
    buffer.write(struct.pack('<f', 0.0))  # distance
    buffer.write(struct.pack('<I', 0))  # direct_control
    buffer.write(struct.pack('<I', vertex.level_vertex_id))  # level_vertex_id
    buffer.write(struct.pack('<I', 0))  # flags
    buffer.write(struct.pack('<H', 0))  # custom_data_size
    buffer.write(struct.pack('<II', 0xFFFFFFFF, 0xFFFFFFFF))  # story_id, spawn_story_id

    # CSE_ALifeGraphPoint fields:
    buffer.write(b'\x00')  # connection_point_name (empty stringZ)
    buffer.write(b'\x00')  # connection_level_name (empty stringZ)
    buffer.write(vertex.vertex_types)  # location types (4 bytes)

    # Go back and write data_size
    data_end = buffer.tell()
    data_size = data_end - data_start - 2
    buffer.seek(data_start)
    buffer.write(struct.pack('<H', data_size))

    return buffer.getvalue()


def parse_spawn_packet(packet: bytes, vertex_index: int) -> Optional[SpawnEntity]:
    """Parse a spawn packet and extract key fields"""
    try:
        offset = 0

        # Size prefix
        if len(packet) < 2:
            return None
        packet_size = struct.unpack_from('<H', packet, offset)[0]
        offset += 2

        # M_SPAWN
        if offset + 2 > len(packet):
            return None
        msg_type = struct.unpack_from('<H', packet, offset)[0]
        if msg_type != 1:
            return None
        offset += 2

        # Section name
        section_name, offset = read_stringz(packet, offset)

        # Entity name
        entity_name, offset = read_stringz(packet, offset)

        # gameid, rp
        offset += 2

        # Position
        if offset + 12 > len(packet):
            return None
        position = struct.unpack_from('<3f', packet, offset)
        offset += 12

        # Angle
        offset += 12

        # respawn, id, parent, phantom
        offset += 8

        # s_flags
        if offset + 2 > len(packet):
            return None
        s_flags = struct.unpack_from('<H', packet, offset)[0]
        offset += 2

        # Version
        version = 0
        if s_flags & 0x20:
            if offset + 2 > len(packet):
                return None
            version = struct.unpack_from('<H', packet, offset)[0]
            offset += 2

        # game_type
        if version > 120:
            offset += 2

        # script_version
        if version > 69:
            offset += 2

        # client_data
        if version > 70:
            if offset + 2 > len(packet):
                return None
            client_size = struct.unpack_from('<H', packet, offset)[0]
            offset += 2 + client_size

        # spawn_id
        spawn_id = 0
        if version > 79:
            if offset + 2 > len(packet):
                return None
            spawn_id = struct.unpack_from('<H', packet, offset)[0]
            offset += 2

        # data_size
        if offset + 2 > len(packet):
            return None
        data_size = struct.unpack_from('<H', packet, offset)[0]
        data_size_offset = offset
        offset += 2

        # STATE data - game_vertex_id
        state_start = offset
        if offset + 2 > len(packet):
            return None
        game_vertex_id = struct.unpack_from('<H', packet, offset)[0]
        offset += 2

        # distance
        offset += 4

        # direct_control
        offset += 4

        # level_vertex_id
        if offset + 4 > len(packet):
            return None
        level_vertex_id = struct.unpack_from('<I', packet, offset)[0]

        # Build comparison data (packet with graph IDs zeroed out)
        comparison = bytearray(packet)
        # Zero out game_vertex_id (2 bytes at state_start)
        struct.pack_into('<H', comparison, state_start, 0)
        # Zero out level_vertex_id (4 bytes at state_start + 10)
        struct.pack_into('<I', comparison, state_start + 10, 0)
        # Zero out spawn_id in header if present
        if version > 79:
            struct.pack_into('<H', comparison, data_size_offset - 2, 0)

        return SpawnEntity(
            spawn_id=spawn_id,
            vertex_index=vertex_index,
            section_name=section_name,
            entity_name=entity_name,
            position=position,
            game_vertex_id=game_vertex_id,
            level_vertex_id=level_vertex_id,
            raw_packet=packet,
            comparison_data=bytes(comparison)
        )

    except Exception as e:
        print(f"  Warning: Failed to parse packet at vertex {vertex_index}: {e}")
        return None


def parse_all_spawn(filepath: Path) -> Tuple[
    Dict[int, LevelInfo], Dict[int, int], List[GameGraphVertex], Dict[tuple, SpawnEntity]]:
    """
    Parse all.spawn and return:
    1. Level definitions
    2. Vertex to level mapping
    3. Game graph vertices (for graph point extraction)
    4. Entities by name
    """
    print(f"\nParsing: {filepath}")

    levels = {}
    vertex_to_level = {}
    graph_vertices = []
    entities = {}

    with open(filepath, 'rb') as f:
        data = f.read()

    print(f"  File size: {len(data):,} bytes")

    # Find chunks
    offset = 0
    game_graph_data = None
    spawn_graph_data = None

    while offset < len(data) - 8:
        chunk_id, chunk_size = struct.unpack_from('<II', data, offset)
        chunk_data = data[offset + 8:offset + 8 + chunk_size]

        if chunk_id == 0:
            print(f"  Found header chunk (0): {chunk_size:,} bytes")
        elif chunk_id == 1:
            spawn_graph_data = chunk_data
            print(f"  Found spawn graph chunk (1): {chunk_size:,} bytes")
        elif chunk_id == 2:
            print(f"  Found artefact spawns chunk (2): {chunk_size:,} bytes")
        elif chunk_id == 3:
            print(f"  Found patrol paths chunk (3): {chunk_size:,} bytes")
        elif chunk_id == 4:
            game_graph_data = chunk_data
            print(f"  Found game graph chunk (4): {chunk_size:,} bytes")

        offset += 8 + chunk_size

    # Parse game graph (now returns vertices too)
    if game_graph_data:
        levels, vertex_to_level, graph_vertices = parse_game_graph(game_graph_data)
    else:
        print("  WARNING: No game graph chunk found!")

    # Parse spawn graph
    if not spawn_graph_data:
        print("  ERROR: No spawn graph chunk found!")
        return levels, vertex_to_level, graph_vertices, entities

    # Parse spawn graph structure
    sg_offset = 0
    vertex_count = 0
    vertices_data = None

    while sg_offset < len(spawn_graph_data) - 8:
        sub_id, sub_size = struct.unpack_from('<II', spawn_graph_data, sg_offset)
        sub_data = spawn_graph_data[sg_offset + 8:sg_offset + 8 + sub_size]

        if sub_id == 0:
            vertex_count = struct.unpack_from('<I', sub_data, 0)[0]
            print(f"  Spawn entity count: {vertex_count:,}")
        elif sub_id == 1:
            vertices_data = sub_data
            print(f"  Spawn vertices chunk: {sub_size:,} bytes")

        sg_offset += 8 + sub_size

    if not vertices_data:
        print("  ERROR: No vertices data found!")
        return levels, vertex_to_level, graph_vertices, entities

    # Parse each spawn vertex
    v_offset = 0
    parsed = 0

    while v_offset < len(vertices_data) - 8:
        v_chunk_id, v_chunk_size = struct.unpack_from('<II', vertices_data, v_offset)
        v_chunk_data = vertices_data[v_offset + 8:v_offset + 8 + v_chunk_size]

        # Parse vertex sub-chunks
        vv_offset = 0
        vertex_id = 0
        spawn_packet = None
        update_packet = None

        while vv_offset < len(v_chunk_data) - 8:
            vv_id, vv_size = struct.unpack_from('<II', v_chunk_data, vv_offset)
            vv_data = v_chunk_data[vv_offset + 8:vv_offset + 8 + vv_size]

            if vv_id == 0:
                vertex_id = struct.unpack_from('<H', vv_data, 0)[0]
            elif vv_id == 1:
                # CServerEntityWrapper - contains M_SPAWN and M_UPDATE
                wrapper_offset = 0
                while wrapper_offset < len(vv_data) - 8:
                    w_id, w_size = struct.unpack_from('<II', vv_data, wrapper_offset)
                    w_data = vv_data[wrapper_offset + 8:wrapper_offset + 8 + w_size]

                    if w_id == 0:  # M_SPAWN packet
                        spawn_packet = w_data
                    elif w_id == 1:  # M_UPDATE packet
                        update_packet = w_data

                    wrapper_offset += 8 + w_size

            vv_offset += 8 + vv_size

        if spawn_packet:
            entity = parse_spawn_packet(spawn_packet, vertex_id)
            if entity:
                entity.update_packet = update_packet  # Attach the M_UPDATE packet
                entities[make_entity_key(entity)] = entity
                parsed += 1

        v_offset += 8 + v_chunk_size

    print(f"  Parsed {parsed:,} spawn entities")
    return levels, vertex_to_level, graph_vertices, entities


def write_level_spawn(entities: List[SpawnEntity], graph_point_packets: List[Tuple[str, bytes]], output_path: Path):
    """
    Write entities and graph points to a level.spawn format file.

    Uses WRAPPER FORMAT so spawn_graph_builder can read both M_SPAWN and M_UPDATE:
    - Outer chunk: chunk_id (u32), chunk_size (u32), wrapper_data
    - wrapper_data contains:
      - Sub-chunk 0: id=0 (u32), size (u32), spawn_packet (with u16 size prefix)
      - Sub-chunk 1: id=1 (u32), size (u32), update_packet (with u16 size prefix) [if present]
    """
    with open(output_path, 'wb') as f:
        chunk_id = 0

        # Write spawn entities in wrapper format
        for entity in entities:
            # Build wrapper data
            wrapper = io.BytesIO()

            # Sub-chunk 0: M_SPAWN (raw_packet already has u16 size prefix)
            spawn_data = entity.raw_packet
            wrapper.write(struct.pack('<II', 0, len(spawn_data)))
            wrapper.write(spawn_data)

            # Sub-chunk 1: M_UPDATE (if present)
            if entity.update_packet is not None:
                update_data = entity.update_packet
                wrapper.write(struct.pack('<II', 1, len(update_data)))
                wrapper.write(update_data)

            # Write outer chunk
            wrapper_data = wrapper.getvalue()
            f.write(struct.pack('<II', chunk_id, len(wrapper_data)))
            f.write(wrapper_data)
            chunk_id += 1

        # Write graph point packets (no update packet, use simple format)
        for name, packet in graph_point_packets:
            # Graph points from create_graph_point_packet don't have size prefix
            # Wrap them in wrapper format for consistency
            wrapper = io.BytesIO()
            spawn_with_prefix = struct.pack('<H', len(packet)) + packet
            wrapper.write(struct.pack('<II', 0, len(spawn_with_prefix)))
            wrapper.write(spawn_with_prefix)

            wrapper_data = wrapper.getvalue()
            f.write(struct.pack('<II', chunk_id, len(wrapper_data)))
            f.write(wrapper_data)
            chunk_id += 1

    entity_count = len(entities)
    gp_count = len(graph_point_packets)
    update_count = sum(1 for e in entities if e.update_packet is not None)
    print(f"  Wrote {entity_count} entities ({update_count} with M_UPDATE) + {gp_count} graph points to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract spawn entities from all.spawn into per-level files',
        epilog='Examples:\n'
               '  %(prog)s all.spawn output/          # Extract ALL entities\n'
               '  %(prog)s all.spawn output/ --original orig.spawn  # Compare mode\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('all_spawn', type=Path,
                        help='Path to all.spawn file to extract from')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for per-level spawn files')
    parser.add_argument('--original', type=Path, default=None,
                        help='Optional original all.spawn for comparison mode (extract only differences)')
    parser.add_argument('--names-ini', type=Path, default=None,
                        help='INI file mapping coordinates to custom graph point names')
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load custom graph point names if provided
    custom_names = {}
    if args.names_ini and args.names_ini.exists():
        custom_names = parse_graph_point_names_ini(args.names_ini)
        total = sum(len(v) for v in custom_names.values())
        print(f"\nLoaded {total} custom graph point names from {args.names_ini}")

    comparison_mode = args.original is not None

    if comparison_mode:
        # --- COMPARISON MODE (existing behavior) ---
        # Parse original (includes game graph for level boundaries AND graph points)
        orig_levels, orig_vertex_to_level, orig_graph_vertices, original_entities = parse_all_spawn(args.original)

        # Parse generated (may not have valid game graph, just need entities)
        _, _, gen_graph_vertices, generated_entities = parse_all_spawn(args.all_spawn)

        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print('=' * 70)

        # Find entities that differ
        only_in_original = []
        only_in_generated = []
        differ = []
        same = []

        all_keys = set(original_entities.keys()) | set(generated_entities.keys())

        for key in all_keys:
            orig = original_entities.get(key)
            gen = generated_entities.get(key)

            if orig and not gen:
                only_in_original.append(orig)
            elif gen and not orig:
                only_in_generated.append(gen)
            elif orig and gen:
                if orig.comparison_data == gen.comparison_data:
                    same.append(key)
                else:
                    differ.append((orig, gen))

        print(f"\nSpawn Entity Results:")
        print(f"  Only in original: {len(only_in_original)}")
        print(f"  Only in generated: {len(only_in_generated)}")
        print(f"  Same (ignoring graph IDs): {len(same)}")
        print(f"  Different: {len(differ)}")

        print(f"\nGraph Point Results:")
        print(f"  Original game graph vertices: {len(orig_graph_vertices)}")
        print(f"  Generated game graph vertices: {len(gen_graph_vertices)}")

        levels = orig_levels
        vertex_to_level = orig_vertex_to_level
        graph_vertices = orig_graph_vertices

        # Entities to extract: only_in_original + differ (use original version)
        extract_entities = list(only_in_original)
        for orig, gen in differ:
            extract_entities.append(orig)

        # Build set of generated graph vertex IDs for comparison
        gen_vertex_ids = {v.vertex_id for v in gen_graph_vertices}
    else:
        # --- SINGLE-SOURCE MODE (extract ALL entities) ---
        levels, vertex_to_level, graph_vertices, all_entities = parse_all_spawn(args.all_spawn)

        print(f"\n{'=' * 70}")
        print("EXTRACTION (single-source mode)")
        print('=' * 70)

        print(f"\nSpawn Entity Results:")
        print(f"  Total entities: {len(all_entities)}")

        print(f"\nGraph Point Results:")
        print(f"  Game graph vertices: {len(graph_vertices)}")

        # Extract ALL entities
        extract_entities = list(all_entities.values())

        gen_vertex_ids = None  # Not used in single-source mode

    # Build level name lookup
    level_id_to_name = {info.level_id: info.name for info in levels.values()}

    # Group entities by level
    entities_by_level: Dict[str, List[SpawnEntity]] = {}
    unmapped_entities = []

    def assign_to_level(entity: SpawnEntity):
        gvid = entity.game_vertex_id
        if gvid == 0xFFFF:
            unmapped_entities.append(entity)
            return
        level_id = vertex_to_level.get(gvid)
        if level_id is None:
            unmapped_entities.append(entity)
            return
        level_name = level_id_to_name.get(level_id, f"level_{level_id}")
        if level_name not in entities_by_level:
            entities_by_level[level_name] = []
        entities_by_level[level_name].append(entity)

    for entity in extract_entities:
        assign_to_level(entity)

    # Group graph points by level
    graph_points_by_level: Dict[str, List[Tuple[str, bytes]]] = {}

    missing_gp_count = 0
    for vertex in graph_vertices:
        level_name = level_id_to_name.get(vertex.level_id, f"level_{vertex.level_id}")

        if level_name not in graph_points_by_level:
            graph_points_by_level[level_name] = []

        # Generate name for this graph point (use custom name if available)
        level_coords = custom_names.get(level_name, {})
        custom_name = find_matching_name(vertex.local_position, level_coords)
        if custom_name:
            gp_name = custom_name
        else:
            gp_name = f"graph_point_{vertex.vertex_id:04d}"
        packet = create_graph_point_packet(vertex, gp_name)
        graph_points_by_level[level_name].append((gp_name, packet))

        if gen_vertex_ids is not None and vertex.vertex_id not in gen_vertex_ids:
            missing_gp_count += 1

    if comparison_mode:
        print(f"  Graph points missing from generated: {missing_gp_count}")

    print(f"\nEntities to extract by level:")
    for level_name in sorted(set(entities_by_level.keys()) | set(graph_points_by_level.keys())):
        ent_count = len(entities_by_level.get(level_name, []))
        gp_count = len(graph_points_by_level.get(level_name, []))
        print(f"  {level_name}: {ent_count} entities, {gp_count} graph points")

    if unmapped_entities:
        print(f"  (unmapped - invalid game_vertex_id): {len(unmapped_entities)} entities")

    # Write per-level spawn files
    print(f"\nWriting level spawn files to {output_dir}/")

    all_levels = sorted(set(entities_by_level.keys()) | set(graph_points_by_level.keys()))

    for level_name in all_levels:
        ents = entities_by_level.get(level_name, [])
        gps = graph_points_by_level.get(level_name, [])

        # Sort entities by game_vertex_id
        ents.sort(key=lambda e: (e.game_vertex_id, e.entity_name))

        output_path = output_dir / f"{level_name}.spawn"
        write_level_spawn(ents, gps, output_path)

    # Write unmapped entities
    if unmapped_entities:
        unmapped_entities.sort(key=lambda e: e.entity_name)
        output_path = output_dir / "unmapped.spawn"
        write_level_spawn(unmapped_entities, [], output_path)
        print(f"  Wrote {len(unmapped_entities)} unmapped entities to {output_path}")

    # Write summary JSON
    if comparison_mode:
        summary = {
            'mode': 'comparison',
            'source_file': str(args.all_spawn),
            'original_file': str(args.original),
            'spawn_entities': {
                'only_in_original': len(only_in_original),
                'only_in_generated': len(only_in_generated),
                'same': len(same),
                'different': len(differ),
            },
            'graph_points': {
                'original_count': len(orig_graph_vertices),
                'generated_count': len(gen_graph_vertices),
                'missing_in_generated': missing_gp_count,
            },
            'levels_from_game_graph': [
                {'level_id': lid, 'name': name}
                for lid, name in sorted(level_id_to_name.items())
            ],
            'extracted_by_level': {
                level: {
                    'entities': [e.entity_name for e in entities_by_level.get(level, [])],
                    'graph_points': len(graph_points_by_level.get(level, []))
                }
                for level in all_levels
            },
            'unmapped_entities': [e.entity_name for e in unmapped_entities]
        }
    else:
        summary = {
            'mode': 'single_source',
            'source_file': str(args.all_spawn),
            'spawn_entities': {
                'total': len(extract_entities),
            },
            'graph_points': {
                'total': len(graph_vertices),
            },
            'levels_from_game_graph': [
                {'level_id': lid, 'name': name}
                for lid, name in sorted(level_id_to_name.items())
            ],
            'extracted_by_level': {
                level: {
                    'entities': [e.entity_name for e in entities_by_level.get(level, [])],
                    'graph_points': len(graph_points_by_level.get(level, []))
                }
                for level in all_levels
            },
            'unmapped_entities': [e.entity_name for e in unmapped_entities]
        }

    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {summary_path}")

    # Write report
    report_path = output_dir / "diff_report.txt"
    with open(report_path, 'w') as f:
        f.write("SPAWN EXTRACTION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Graph Points: {len(graph_vertices)} extracted from game graph\n\n")

        if comparison_mode:
            f.write(f"Mode: comparison\n")
            f.write(f"Original: {args.original}\n")
            f.write(f"Generated: {args.all_spawn}\n\n")

            f.write(f"Only in original ({len(only_in_original)}):\n")
            for e in sorted(only_in_original, key=lambda x: x.entity_name):
                level_id = vertex_to_level.get(e.game_vertex_id, -1)
                level_name = level_id_to_name.get(level_id, "unknown")
                f.write(f"  {e.entity_name} (section={e.section_name}, gvid={e.game_vertex_id}, level={level_name})\n")

            f.write(f"\nOnly in generated ({len(only_in_generated)}):\n")
            for e in sorted(only_in_generated, key=lambda x: x.entity_name):
                f.write(f"  {e.entity_name} (section={e.section_name}, gvid={e.game_vertex_id})\n")

            f.write(f"\nDifferent ({len(differ)}):\n")
            for orig, gen in sorted(differ, key=lambda x: x[0].entity_name):
                level_id = vertex_to_level.get(orig.game_vertex_id, -1)
                level_name = level_id_to_name.get(level_id, "unknown")
                f.write(f"  {orig.entity_name} (level={level_name}):\n")
                f.write(
                    f"    Original: {len(orig.raw_packet)} bytes, gvid={orig.game_vertex_id}, lvid={orig.level_vertex_id}\n")
                f.write(
                    f"    Generated: {len(gen.raw_packet)} bytes, gvid={gen.game_vertex_id}, lvid={gen.level_vertex_id}\n")
        else:
            f.write(f"Mode: single_source\n")
            f.write(f"Source: {args.all_spawn}\n\n")

            f.write(f"Total entities extracted: {len(extract_entities)}\n\n")
            f.write(f"Entities by level:\n")
            for level_name in all_levels:
                ent_count = len(entities_by_level.get(level_name, []))
                gp_count = len(graph_points_by_level.get(level_name, []))
                f.write(f"  {level_name}: {ent_count} entities, {gp_count} graph points\n")

            if unmapped_entities:
                f.write(f"\nUnmapped entities ({len(unmapped_entities)}):\n")
                for e in unmapped_entities:
                    f.write(f"  {e.entity_name} (section={e.section_name}, gvid={e.game_vertex_id})\n")

    print(f"Wrote report to {report_path}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print('=' * 70)


if __name__ == '__main__':
    main()