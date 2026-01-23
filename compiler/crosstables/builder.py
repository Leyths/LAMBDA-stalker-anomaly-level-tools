#!/usr/bin/env python3
"""
Cross Table Builder

Builds cross tables (.gct files) that map level-local vertex IDs to global game graph vertex IDs.

Usage (CLI):
    python builder.py --level-ai level.ai --level-spawn level.spawn --output level.gct
    python builder.py --level-ai level.ai --level-spawn level.spawn --original-spawn old.spawn --output level.gct
"""

import sys
import struct
import numpy as np
from pathlib import Path
from typing import List, Optional
import time
import argparse

from utils import log, logDebug, logError, generate_guid
from constants import XRAI_CURRENT_VERSION

from .data_types import GraphPoint, CrossTableCell
from .level_graph_navigator import LevelGraphNavigator
from .graph_point_parser import extract_and_merge_graph_points


class CrossTableBuilder:
    """Build cross table using distance matrix algorithm"""

    def __init__(self, graph_points: List[GraphPoint], level_graph: LevelGraphNavigator):
        self.graph_points = graph_points
        self.level_graph = level_graph
        self.vertex_count = level_graph.header['vertex_count']
        self.cell_size = level_graph.header['cell_size']

    def build(self) -> List[CrossTableCell]:
        log(f"\nBuilding cross table...")
        log(f"  Game vertices: {len(self.graph_points)}")
        log(f"  Level vertices: {self.vertex_count:,}")

        # Get all starting vertices
        start_vertices = [gp.level_vertex_id for gp in self.graph_points]

        log(f"\n  Calculating distances (single multi-source BFS)...")
        start = time.time()

        # Single BFS from all graph points simultaneously
        assignments, distances_edges = self.level_graph.multi_source_bfs_distances(start_vertices)

        elapsed = time.time() - start
        log(f"  BFS complete in {elapsed:.1f}s")

        # Convert edge distances to meters
        INFINITY_EDGES = np.iinfo(np.uint32).max
        INFINITY = np.finfo(np.float32).max
        distances_meters = np.where(
            distances_edges == INFINITY_EDGES,
            INFINITY,
            distances_edges.astype(np.float32) * self.cell_size
        )

        # Build cross table
        log(f"\n  Building cross table cells...")
        cross_table = []
        for level_vertex_id in range(self.vertex_count):
            cell = CrossTableCell(
                game_vertex_id=int(assignments[level_vertex_id]),
                distance=float(distances_meters[level_vertex_id])
            )
            cross_table.append(cell)

        return cross_table


def write_cross_table_gct(output_path: str,
                          level_guid: bytes,
                          game_guid: bytes,
                          level_vertex_count: int,
                          game_vertex_count: int,
                          cells: List[CrossTableCell]):
    """Write binary level.gct file"""

    # Handle None or invalid level GUID
    if not level_guid or len(level_guid) != 16:
        level_guid = generate_guid(
            "level_ai",
            output_path,
            level_vertex_count
        )

    if not game_guid or len(game_guid) != 16:
        # Generate deterministic GUID based on cross table content
        game_guid = generate_guid(
            "cross_table",
            level_guid,
            level_vertex_count,
            game_vertex_count
        )

    with open(output_path, 'wb') as f:
        # Chunk 0: Header (version chunk)
        chunk_id = 0xFFFF
        header_data = struct.pack(
            '<I I I 16s 16s',
            XRAI_CURRENT_VERSION,
            level_vertex_count,
            game_vertex_count,
            level_guid,
            game_guid
        )
        f.write(struct.pack('<I I', chunk_id, len(header_data)))
        f.write(header_data)

        # Chunk 1: Data
        chunk_id = 1
        cells_data = bytearray()
        for cell in cells:
            cells_data.extend(struct.pack('<H f', cell.game_vertex_id, cell.distance))

        f.write(struct.pack('<I I', chunk_id, len(cells_data)))
        f.write(cells_data)


def build_cross_table_for_level(
    level_ai_path: Path,
    level_spawn_path: Path,
    output_path: Path,
    original_spawn_path: Optional[Path] = None
) -> bool:
    """
    Build cross table for a single level.

    This is the primary API for building cross tables programmatically.
    Can be called directly instead of via subprocess.

    Args:
        level_ai_path: Path to level.ai file
        level_spawn_path: Path to level.spawn file (binary)
        output_path: Output path for .gct file
        original_spawn_path: Optional path to original spawn file for merging

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load level graph
        log(f"    Loading level graph...")
        level_graph = LevelGraphNavigator(str(level_ai_path))
        logDebug(f"      Loaded: {level_graph.header['vertex_count']:,} vertices")

        # Extract graph points (merged from both sources)
        logDebug(f"    Extracting graph points...")
        graph_points = extract_and_merge_graph_points(level_spawn_path, original_spawn_path)

        if not graph_points:
            logError(f"    Error: No graph points found!")
            return False

        # Assign level vertices - ALWAYS recompute from current level.ai
        logDebug(f"    Assigning level vertices to graph points...")
        for gp in graph_points:
            gp.level_vertex_id = level_graph.find_nearest_vertex(gp.position)
            logDebug(f"      {gp.name} -> level vertex {gp.level_vertex_id}")

        # Build cross table
        log(f"    Building cross table ({len(graph_points)} game vertices, {level_graph.header['vertex_count']:,} level vertices)...")
        builder = CrossTableBuilder(graph_points, level_graph)
        cross_table = builder.build()

        # Write output
        logDebug(f"    Writing {output_path}...")
        write_cross_table_gct(
            str(output_path),
            level_graph.header['guid'],
            b'\x00' * 16,  # Generate new GUID
            level_graph.header['vertex_count'],
            len(graph_points),
            cross_table
        )

        file_size = output_path.stat().st_size / 1024 / 1024
        log(f"    Wrote {output_path.name} ({file_size:.1f} MB)")

        return True

    except Exception as e:
        logError(f"    Error building cross table: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Build cross table from level data')
    parser.add_argument('--level-ai', required=True, help='Path to level.ai file')
    parser.add_argument('--level-spawn', required=True, help='Path to level.spawn file (binary)')
    parser.add_argument('--original-spawn', help='Path to original spawn file for merging (optional)')
    parser.add_argument('--output', required=True, help='Output level.gct file')

    args = parser.parse_args()

    log("=" * 70)
    log("Cross Table Builder")
    log("=" * 70)

    # Load level graph
    log(f"\n[1/5] Loading level graph from {args.level_ai}...")
    level_graph = LevelGraphNavigator(args.level_ai)
    log(f"  Loaded: {level_graph.header['vertex_count']:,} vertices")
    log(f"  Cell size: {level_graph.header['cell_size']:.3f} meters")

    # Extract graph points (merged from both sources)
    log(f"\n[2/5] Extracting graph points...")
    level_spawn_path = Path(args.level_spawn)
    original_spawn_path = Path(args.original_spawn) if args.original_spawn else None

    graph_points = extract_and_merge_graph_points(level_spawn_path, original_spawn_path)

    if not graph_points:
        logError("\nError: No graph points found!")
        sys.exit(1)

    # Assign level vertices - ALWAYS recompute from current level.ai
    # Values from spawn files may be for different level.ai files and would be invalid
    log(f"\n[3/5] Assigning level vertices to graph points...")
    for gp in graph_points:
        # Always find the nearest vertex in the current level.ai
        gp.level_vertex_id = level_graph.find_nearest_vertex(gp.position)
        log(f"  {gp.name} -> level vertex {gp.level_vertex_id}")

    # Build cross table
    log(f"\n[4/5] Building cross table...")
    builder = CrossTableBuilder(graph_points, level_graph)
    cross_table = builder.build()

    # Write output
    log(f"\n[5/5] Writing {args.output}...")
    write_cross_table_gct(
        args.output,
        level_graph.header['guid'],
        b'\x00' * 16,  # Generate new GUID
        level_graph.header['vertex_count'],
        len(graph_points),
        cross_table
    )

    file_size = Path(args.output).stat().st_size / 1024 / 1024
    log(f"  Wrote {file_size:.1f} MB")

    log("\n" + "=" * 70)
    log("COMPLETE!")
    log("=" * 70)


if __name__ == '__main__':
    main()
