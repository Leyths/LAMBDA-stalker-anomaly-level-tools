#!/usr/bin/env python3
"""
Patrol Path Extractor

Extracts patrol paths from level files and merges them for all.spawn.

Patrol paths are typically stored in:
1. level.game (compiled level file) - contains patrol path data
2. Or generated during level compilation from waypoints

This extractor reads patrol paths from compiled level files if available.
"""

import struct
import io
from pathlib import Path
from typing import List, Dict, Tuple, TYPE_CHECKING

from utils import log, logDebug, logWarning, logError, write_chunk

if TYPE_CHECKING:
    from graph import GameGraph


def extract_patrol_paths_from_level(level_config,
                                    game_graph: 'GameGraph' = None) -> Dict[str, bytes]:
    """
    Extract patrol paths from a level directory

    Merges patrol paths from:
    1. level.game (if exists) - new/compiled patrol paths
    2. original_patrols file (if provided) - extracted from original all.spawn

    Args:
        level_config: LevelConfig object with level metadata
        game_graph: GameGraph object for GVID resolution

    Returns:
        Dictionary of {patrol_name: patrol_data}
    """
    from .read_extracted_patrols import read_extracted_patrols
    from remapping import validate_and_remap_patrols

    level_name = level_config.name
    level_dir = Path(level_config.path)

    if not level_dir.exists():
        logError(f"      Level directory not found: {level_dir}")
        return {}

    # Resolve original_patrols path if provided
    original_patrols_path = None
    if level_config.original_patrols:
        original_patrols_path = Path(level_config.original_patrols)
        if not original_patrols_path.exists():
            # Skip logging for fake_start as this is expected
            if level_name != "fake_start":
                logError(f"      Original patrols file not found: {original_patrols_path}")
            original_patrols_path = None

    new_patrols = {}
    original_patrols = {}

    # 1. Load original patrols if provided
    if original_patrols_path and original_patrols_path.exists():
        logDebug(f"      Loading original patrols from {original_patrols_path.name}")
        original_patrols = read_extracted_patrols(original_patrols_path)
        logDebug(f"        Loaded {len(original_patrols)} original patrol paths")

    # 2. Check for level.game file (contains compiled level data)
    level_game = level_dir / "level.game"
    if level_game.exists():
        log(f"      Found level.game")
        try:
            # Get paths for vertex lookups
            level_ai = level_dir / "level.ai"
            cross_table = Path(f"../.tmp/{level_name}.gct")

            if not level_ai.exists():
                logError(f"      level.ai not found: {level_ai}")
            if not cross_table.exists():
                logError(f"      cross_table not found: {cross_table}")

            new_patrols = _extract_from_level_game(level_game, level_ai, cross_table)
            log(f"        Extracted {len(new_patrols)} patrol paths from level.game")
        except Exception as e:
            logWarning(f"      Could not extract patrols from {level_game}: {e}")
            import traceback
            traceback.print_exc()
    else:
        logError(f"      level.game not found: {level_game}")

    # 3. Merge if we have both sources
    if original_patrols and new_patrols:
        logDebug(f"      Merging original and new patrol paths...")

        # Import and use the old merge function with remapping integration
        from .patrol_path_merger import merge_patrol_paths_with_game_graph
        merged_patrols = merge_patrol_paths_with_game_graph(
            new_patrols, original_patrols, game_graph, level_name
        )
        logDebug(f"      Result: {len(merged_patrols)} total patrol paths")
        return merged_patrols

    # 4. Return whichever we have
    if original_patrols:
        logDebug(f"      Using {len(original_patrols)} original patrol paths")
        # Update IDs just like we do when merging - this was previously a bug
        # where original patrols were returned without updating GVIDs!
        if game_graph:
            level_ai = game_graph.get_level_ai_for_level(level_name)
            cross_table = game_graph.get_cross_table_for_level(level_name)

            if level_ai is None:
                logError(f"      Cannot update patrol IDs: level.ai not cached for {level_name}")
                return original_patrols
            if cross_table is None:
                logError(f"      Cannot update patrol IDs: cross table not cached for {level_name}")
                return original_patrols

            return validate_and_remap_patrols(original_patrols, game_graph, level_name)
        else:
            logError(f"      Cannot update patrol IDs: GameGraph not provided")
            return original_patrols

    if new_patrols:
        logDebug(f"      Using {len(new_patrols)} new patrol paths")
        return new_patrols

    # fake_start is a quirk of the engine and has no patrol paths by design
    if level_name != "fake_start":
        logError(f"      No patrol paths found for level: {level_name}")
    return {}


def _extract_from_level_game(level_game_path: Path,
                             level_ai_path: Path,
                             cross_table_path: Path) -> Dict[str, bytes]:
    """
    Extract patrol paths from level.game file

    level.game contains waypoints in chunks 0x1000+
    """
    from levels import parse_level_game
    from converters.waypoint import convert_wayobjects_to_patrol_paths

    waypoints = parse_level_game(level_game_path)

    # Extract patrol paths from chunk 0x1000 (wtPatrolPath)
    if 0x1000 in waypoints:
        return convert_wayobjects_to_patrol_paths(
            waypoints[0x1000],
            level_ai_path,
            cross_table_path
        )

    return {}


def merge_patrol_paths(level_configs: List = None,
                       game_graph: 'GameGraph' = None) -> bytes:
    """
    Merge patrol paths from multiple sources

    Args:
        level_configs: List of LevelConfig objects with level info and original_patrols paths
        game_graph: GameGraph object for GVID resolution

    Returns:
        Binary patrol path data for all.spawn chunk 3
    """
    log("  Extracting patrol paths from levels...")

    all_patrols = {}

    if not level_configs:
        logError("    No level configs provided")
        return _build_empty_patrol_storage()

    for level_config in level_configs:
        log(f"    Processing {level_config.name}...")

        # Extract patrols (merging new and original if both exist)
        patrols = extract_patrol_paths_from_level(
            level_config,
            game_graph
        )

        if patrols:
            log(f"      Found {len(patrols)} patrol paths total")
            all_patrols.update(patrols)

    if not all_patrols:
        logError("    No patrol paths found in any level")
        return _build_empty_patrol_storage()

    log(f"    Total: {len(all_patrols)} patrol paths from {len(level_configs)} levels")
    return _build_patrol_storage(all_patrols)


def _build_empty_patrol_storage() -> bytes:
    """Build empty patrol path storage"""
    buffer = io.BytesIO()

    # Chunk 0: patrol count = 0
    write_chunk(buffer, 0, struct.pack('<I', 0))

    return buffer.getvalue()


def _build_patrol_storage(patrols: Dict[str, bytes]) -> bytes:
    """
    Build patrol path storage from patrol data

    Format:
    - Chunk 0: patrol_count (u32)
    - Chunk 1: Patrol paths
      - For each patrol i:
        - Sub-chunk 0: name (stringZ)
        - Sub-chunk 1: CPatrolPath data
    """
    buffer = io.BytesIO()

    # Chunk 0: patrol count
    count_buffer = io.BytesIO()
    count_buffer.write(struct.pack('<I', len(patrols)))
    write_chunk(buffer, 0, count_buffer.getvalue())

    # Chunk 1: Patrol paths
    paths_buffer = io.BytesIO()

    for i, (name, data) in enumerate(sorted(patrols.items())):
        path_buffer = io.BytesIO()

        # Sub-chunk 0: name
        name_buffer = io.BytesIO()
        name_buffer.write(name.encode('utf-8') + b'\x00')
        write_chunk(path_buffer, 0, name_buffer.getvalue())

        # Sub-chunk 1: patrol data
        write_chunk(path_buffer, 1, data)

        write_chunk(paths_buffer, i, path_buffer.getvalue())

    write_chunk(buffer, 1, paths_buffer.getvalue())

    return buffer.getvalue()


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python patrol_path_extractor.py level_dir1 [level_dir2 ...]")
        sys.exit(1)

    level_dirs = [Path(p) for p in sys.argv[1:]]
    data = merge_patrol_paths(level_dirs)

    print(f"\nGenerated {len(data)} bytes of patrol path data")