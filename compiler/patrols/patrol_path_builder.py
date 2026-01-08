#!/usr/bin/env python3
"""
Patrol Path Builder

Builds chunk 3 (patrol paths) for all.spawn.

Extracts patrol paths from level files if available, otherwise creates empty storage.
"""

import struct
import io
from pathlib import Path
from typing import List, TYPE_CHECKING

from .patrol_path_extractor import merge_patrol_paths

if TYPE_CHECKING:
    from graph import GameGraph


def build_patrol_paths(level_configs: List = None, base_path: Path = Path('.'),
                       game_graph: 'GameGraph' = None) -> bytes:
    """
    Build patrol paths chunk

    Attempts to extract patrol paths from level files and original_patrols files.
    Merges and deduplicates patrol paths, updating graph IDs.

    Args:
        level_configs: List of LevelConfig objects
        base_path: Base path for resolving relative paths
        game_graph: GameGraph object for GVID resolution

    Returns:
        Binary patrol paths data (chunk 3)
    """
    print("  Building patrol paths...")

    # Extract and merge from all sources
    return merge_patrol_paths(level_configs, base_path, game_graph)


if __name__ == '__main__':
    # Test
    data = build_patrol_paths([])
    print(f"Generated {len(data)} bytes")