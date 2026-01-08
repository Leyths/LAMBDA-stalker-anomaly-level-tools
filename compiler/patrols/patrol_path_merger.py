#!/usr/bin/env python3
"""
Patrol Path Merger

Merges patrol paths from multiple sources (level.game and original .patrols files).
Uses the remapping module for GVID resolution.
"""

from typing import Dict


def merge_patrol_paths_with_game_graph(new_patrols: Dict[str, bytes],
                                       original_patrols: Dict[str, bytes],
                                       game_graph,
                                       level_name: str) -> Dict[str, bytes]:
    """
    Merge new and original patrol paths using GameGraph for GVID resolution.

    This is a wrapper that uses the new remapping module with GameGraph.

    Args:
        new_patrols: Patrol paths from level.game (if available)
        original_patrols: Patrol paths from extracted .patrols file
        game_graph: GameGraph object for GVID resolution
        level_name: Name of the level

    Returns:
        Merged dict of {patrol_name: updated_patrol_data}
    """
    from remapping import validate_and_remap_patrols

    print(f"      Merging patrols: {len(new_patrols)} new, {len(original_patrols)} original")

    # Validate and update original patrols (they take precedence)
    validated_original = validate_and_remap_patrols(
        original_patrols, game_graph, level_name
    )

    # Validate and update new patrols
    validated_new = validate_and_remap_patrols(
        new_patrols, game_graph, level_name
    )

    # Merge: original takes precedence
    merged = dict(validated_original)

    # Add new patrols (only if not already in original)
    added_new = 0
    for name, patrol_data in validated_new.items():
        if name not in merged:
            merged[name] = patrol_data
            added_new += 1

    if added_new > 0:
        print(f"        Added {added_new} new patrols not in original")

    duplicates = len(validated_new) - added_new
    if duplicates > 0:
        print(f"        Skipped {duplicates} duplicates (using original versions)")

    return merged


if __name__ == '__main__':
    print("Patrol Path Merger Module")
    print("Use this from patrol_path_extractor.py or patrol_path_builder.py")
