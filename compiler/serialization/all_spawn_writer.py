#!/usr/bin/env python3
"""
All Spawn Builder

Assembles level.spawn files and game graph into the final all.spawn file.

Strategy:
1. Build spawn graph from level.spawn files (chunk 1)
2. Build patrol paths (chunk 3)
3. Add game graph (chunk 4)
4. Create header (chunk 0)
5. Write all chunks to output file

Supports:
- Merging with original spawn data from extracted files
- Blacklist to exclude specific entities
"""

import struct
from pathlib import Path
from typing import List

from utils import log
from .header_serializer import create_header


def build_all_spawn(game_graph_data: bytes,
                    game_graph_guid: bytes,
                    level_spawn_paths: List[Path],
                    level_count: int,
                    output_path: Path,
                    level_configs: List = None,
                    base_path: Path = None,
                    blacklist_path: Path = None,
                    game_graph = None):
    """
    Build minimal all.spawn file with game graph

    Args:
        game_graph_data: Binary game graph (chunk 4)
        game_graph_guid: GUID from game graph
        level_spawn_paths: Paths to level.spawn files
        level_count: Number of levels
        output_path: Output file path
        level_configs: Level configuration objects (includes per-level original_spawn paths)
        base_path: Base path for resolving relative paths
        blacklist_path: Path to spawn_blacklist.ini file
        game_graph: GameGraph object for GVID resolution
    """
    log("\n" + "=" * 70)
    log("Building all.spawn")
    log("=" * 70)

    from spawn_graph import build_spawn_graph
    from patrols import build_patrol_paths

    # Create chunks
    import io

    # Resolve base_path
    if base_path is None:
        base_path = Path('.')

    # Chunk 1: Spawn graph (build from level.spawn files with ID resolution)
    log("\nBuilding spawn graph...")
    log("  Graph ID resolution: ENABLED")
    log("  Per-level original_spawn: from levels.ini")
    if blacklist_path:
        log(f"  Blacklist: {blacklist_path}")

    (spawn_graph, spawn_count) = build_spawn_graph(
        level_configs=level_configs,
        base_path=base_path,
        blacklist_path=blacklist_path,
        game_graph=game_graph
    )

    # Chunk 2: Empty artefacts
    log("Creating artefact spawns...")
    artefact_buffer = io.BytesIO()
    artefact_buffer.write(struct.pack('<I', 0))  # count = 0
    artefacts = artefact_buffer.getvalue()

    # Chunk 3: Patrol paths
    log("Building patrol paths...")
    patrols = build_patrol_paths(level_configs, base_path, game_graph)

    # Create header
    log("Creating header...")
    header = create_header(game_graph_guid, spawn_count, level_count)

    # Write all.spawn
    log(f"\nWriting all.spawn to {output_path}...")

    with open(output_path, 'wb') as f:
        # Chunk 0: Header
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', len(header)))
        f.write(header)
        log(f"  Chunk 0 (header): {len(header):,} bytes")

        # Chunk 1: Spawn graph
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', len(spawn_graph)))
        f.write(spawn_graph)
        log(f"  Chunk 1 (spawn graph): {len(spawn_graph):,} bytes - COMPLETE")

        # Chunk 2: Artefacts
        f.write(struct.pack('<I', 2))
        f.write(struct.pack('<I', len(artefacts)))
        f.write(artefacts)
        log(f"  Chunk 2 (artefacts): {len(artefacts):,} bytes - EMPTY")

        # Chunk 3: Patrols
        f.write(struct.pack('<I', 3))
        f.write(struct.pack('<I', len(patrols)))
        f.write(patrols)
        log(f"  Chunk 3 (patrols): {len(patrols):,} bytes - MINIMAL")

        # Chunk 4: Game graph
        f.write(struct.pack('<I', 4))
        f.write(struct.pack('<I', len(game_graph_data)))
        f.write(game_graph_data)
        log(f"  Chunk 4 (game graph): {len(game_graph_data):,} bytes - COMPLETE")

    file_size = output_path.stat().st_size / 1024 / 1024
    log(f"\nAll.spawn written: {file_size:.2f} MB")
    log(f"  Total chunks: 5")
    log(f"\n  Spawn graph: COMPLETE - {len(level_spawn_paths)} levels merged")


if __name__ == '__main__':
    import sys

    log("Test: all_spawn_builder")