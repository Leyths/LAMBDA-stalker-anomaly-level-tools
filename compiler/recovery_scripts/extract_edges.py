#!/usr/bin/env python3
"""
Extract edges from a compiled game graph (all.spawn chunk 4).

Edges define connectivity between game vertices and are essential for AI pathfinding.
This extracts them from Anomaly's all.spawn so they can be reused when building a new spawn.

Edges are stored by POSITION (not vertex ID) for robustness - this allows the edge data
to work even if vertex ordering changes between builds.

Output: Per-level files named <level_name>.edges.json with format:
{
    "level_name": "k00_marsh",
    "level_id": 1,
    "edge_count": N,
    "edges": [
        {
            "source_x": float, "source_y": float, "source_z": float,
            "target_x": float, "target_y": float, "target_z": float,
            "distance": float,
            "target_level": int
        },
        ...
    ]
}

Usage:
    python extract_edges.py <all.spawn> <output_dir>
"""

import struct
import json
import sys
from pathlib import Path

# Add parent directory to path for parsers import
sys.path.insert(0, str(Path(__file__).parent.parent))
from parsers import read_stringz, GameGraphParser, ChunkReader


def extract_game_graph_chunk(all_spawn_path: Path) -> bytes:
    """Extract chunk 4 (game graph) from all.spawn"""
    with open(all_spawn_path, 'rb') as f:
        data = f.read()

    reader = ChunkReader(data)
    for chunk in reader:
        if chunk.chunk_id == 4:
            return chunk.data

    raise ValueError("Game graph chunk (4) not found in all.spawn")


def parse_game_graph_edges(game_graph_data: bytes) -> dict:
    """
    Parse game graph header and edges, returning per-level edge data.

    Returns dict: {level_id: {"level_name": str, "level_id": int, "edges": [...]}}
    """
    offset = 0

    # Header
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

    print(f"Game Graph Header:")
    print(f"  Version: {version}")
    print(f"  Vertices: {vertex_count}")
    print(f"  Edges: {edge_count}")
    print(f"  Death points: {death_point_count}")
    print(f"  Levels: {level_count}")

    # Parse level info to get level names
    level_names = {}  # level_id -> level_name
    for i in range(level_count):
        level_name, offset = read_stringz(game_graph_data, offset)

        # Offset (3 floats)
        offset += 12

        # Level ID (u8)
        level_id = struct.unpack_from('<B', game_graph_data, offset)[0]
        offset += 1

        # Section name (stringZ)
        _, offset = read_stringz(game_graph_data, offset)

        # GUID (16 bytes)
        offset += 16

        level_names[level_id] = level_name
        print(f"  Level {level_id}: {level_name}")

    vertices_start = offset
    print(f"  Vertices start at offset: {vertices_start}")

    # Parse ALL vertex data (we need positions and level IDs)
    VERTEX_SIZE = 42
    vertex_data = []  # [(local_pos, global_pos, level_id), ...]

    for v_id in range(vertex_count):
        v_offset = vertices_start + v_id * VERTEX_SIZE

        # Local position (12 bytes)
        local_pos = struct.unpack_from('<3f', game_graph_data, v_offset)

        # Global position (12 bytes)
        global_pos = struct.unpack_from('<3f', game_graph_data, v_offset + 12)

        # level_id from packed field at offset 24
        packed = struct.unpack_from('<I', game_graph_data, v_offset + 24)[0]
        level_id = packed & 0xFF

        # edge_offset (u32 at offset 32) and neighbour_count (u8 at offset 40)
        edge_offset = struct.unpack_from('<I', game_graph_data, v_offset + 32)[0]
        neighbour_count = struct.unpack_from('<B', game_graph_data, v_offset + 40)[0]

        vertex_data.append({
            'local_pos': local_pos,
            'global_pos': global_pos,
            'level_id': level_id,
            'edge_offset': edge_offset,
            'neighbour_count': neighbour_count
        })

    # Parse edges and group by source level
    EDGE_SIZE = 6
    edges_by_level = {}  # level_id -> list of edges

    for v_id, v_data in enumerate(vertex_data):
        actual_edge_offset = vertices_start + v_data['edge_offset']

        source_pos = v_data['local_pos']
        source_level = v_data['level_id']

        if source_level not in edges_by_level:
            edges_by_level[source_level] = []

        for n in range(v_data['neighbour_count']):
            e_offset = actual_edge_offset + n * EDGE_SIZE

            target_vertex_id = struct.unpack_from('<H', game_graph_data, e_offset)[0]
            distance = struct.unpack_from('<f', game_graph_data, e_offset + 2)[0]

            if target_vertex_id < len(vertex_data):
                target_data = vertex_data[target_vertex_id]
                target_pos = target_data['local_pos']
                target_level_id = target_data['level_id']

                # Use level name instead of ID for target_level
                target_level_name = level_names.get(target_level_id, f"level_{target_level_id}")

                edges_by_level[source_level].append({
                    "source_x": round(source_pos[0], 4),
                    "source_y": round(source_pos[1], 4),
                    "source_z": round(source_pos[2], 4),
                    "target_x": round(target_pos[0], 4),
                    "target_y": round(target_pos[1], 4),
                    "target_z": round(target_pos[2], 4),
                    "distance": round(distance, 4),
                    "target_level": target_level_name
                })

    # Build result dict
    result = {}
    total_edges = 0
    total_intra = 0
    total_inter = 0

    for level_id, edges in edges_by_level.items():
        level_name = level_names.get(level_id, f"level_{level_id}")
        # Count intra-level edges (target_level name matches source level name)
        intra = sum(1 for e in edges if e['target_level'] == level_name)
        inter = len(edges) - intra

        result[level_id] = {
            "level_name": level_name,
            "level_id": level_id,
            "edge_count": len(edges),
            "intra_level_edges": intra,
            "inter_level_edges": inter,
            "edges": edges
        }

        total_edges += len(edges)
        total_intra += intra
        total_inter += inter

    print(f"\nEdge Analysis:")
    print(f"  Total edges: {total_edges}")
    print(f"  Intra-level (within same level): {total_intra}")
    print(f"  Inter-level (between levels): {total_inter}")

    print(f"\nEdges per level:")
    for level_id in sorted(result.keys()):
        data = result[level_id]
        print(
            f"  {data['level_name']}: {data['edge_count']} edges ({data['intra_level_edges']} intra, {data['inter_level_edges']} inter)")

    return result, level_names


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_edges.py <all.spawn> <output_dir>")
        print("\nExtracts edge data from a compiled game graph into per-level files.")
        print("Output files are named <level_name>.edges.json")
        print("\nAdd to levels.ini:")
        print("  original_edges = extractedanomalyspawns/<level_name>.edges.json")
        sys.exit(1)

    all_spawn_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting edges from: {all_spawn_path}")
    print(f"Output directory: {output_dir}")

    # Extract game graph chunk
    game_graph_data = extract_game_graph_chunk(all_spawn_path)
    print(f"Game graph chunk size: {len(game_graph_data):,} bytes")

    # Parse edges
    edges_by_level, level_names = parse_game_graph_edges(game_graph_data)

    # Write per-level files
    print(f"\nWriting per-level edge files...")
    for level_id, data in edges_by_level.items():
        level_name = data['level_name']
        output_path = output_dir / f"{level_name}.edges.json"

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  {output_path.name}: {data['edge_count']} edges")

    print(f"\nDone! Add these lines to levels.ini for each level:")
    print(f"  original_edges = {output_dir}/<level_name>.edges.json")


if __name__ == '__main__':
    main()