#!/usr/bin/env python3
"""
Node Graph Inspector - Main Entry Point

A tool for inspecting navigation mesh node graphs (read-only).
Level list is derived from the all.spawn game graph, not levels.ini.
"""
import sys
import os
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
compiler_root = os.path.join(project_root, "compiler")

sys.path.insert(0, compiler_root)
from parsers import GameGraphParser, LevelAIParser

# Remove compiler's utils from module cache so visualiser's utils can be loaded
if 'utils' in sys.modules:
    del sys.modules['utils']

# Now add visualiser directory to Python path for UI imports
sys.path.insert(0, script_dir)


def _resolve_level_ai_path(level_name, levels_dir):
    """Find level.ai for a level in the levels directory."""
    ai_path = os.path.join(levels_dir, level_name, "level.ai")
    if os.path.exists(ai_path):
        return ai_path
    return None


def _validate_guid(ai_path, all_spawn_path, level_id):
    """Check level.ai GUID matches the all.spawn cross table. Warns on mismatch."""
    try:
        ai_parser = LevelAIParser(ai_path, build_adjacency=False)
        ai_guid = ai_parser.guid

        gg = GameGraphParser.from_all_spawn(Path(all_spawn_path))
        cross_table = gg.get_cross_table_for_level(level_id)

        if cross_table and 'level_guid' in cross_table:
            ct_guid = cross_table['level_guid']
            if ai_guid != ct_guid:
                print(f"\nWarning: level.ai GUID does not match the cross table in all.spawn.")
                print(f"  level.ai:     {ai_guid.hex()}")
                print(f"  cross table:  {ct_guid.hex()}")
                print(f"  Vertex IDs may not correspond correctly.\n")
    except Exception:
        pass


def select_level(all_spawn_path, levels_dir):
    """Show level selection menu and return the selected level.ai path and level_id."""
    if not os.path.exists(all_spawn_path):
        print(f"Error: all.spawn not found at {all_spawn_path}")
        sys.exit(1)

    try:
        gg = GameGraphParser.from_all_spawn(Path(all_spawn_path))
        levels = gg.get_levels()
    except Exception as e:
        print(f"Error parsing all.spawn: {e}")
        sys.exit(1)

    print("=" * 60)
    print("LEVEL SELECTION")
    print("=" * 60)
    print()

    # Build list of all levels from the all.spawn game graph
    all_levels = []
    for level_id in sorted(levels.keys()):
        level = levels[level_id]
        ai_path = _resolve_level_ai_path(level.name, levels_dir)
        has_ai = ai_path is not None
        all_levels.append((level.name, level_id, ai_path, has_ai))

    if not all_levels:
        print("No levels found in all.spawn!")
        sys.exit(1)

    # Display menu - show all levels, mark those without level.ai
    for i, (name, lid, ai_path, has_ai) in enumerate(all_levels, 1):
        marker = "" if has_ai else " (no level.ai)"
        print(f"  {i:2d}. {name}{marker}")

    print()
    print("  0. Exit")
    print()

    # Get selection
    while True:
        try:
            choice = input("Select level (number): ").strip()
            if choice == "0" or choice.lower() == "q":
                print("Exiting.")
                sys.exit(0)

            idx = int(choice) - 1
            if 0 <= idx < len(all_levels):
                name, level_id, ai_path, has_ai = all_levels[idx]
                if not has_ai:
                    print(f"  Error: level.ai not found for '{name}'")
                    print(f"  Checked: {os.path.join(levels_dir, name, 'level.ai')}")
                    continue
                return ai_path, level_id
            else:
                print(f"Please enter a number between 1 and {len(all_levels)}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)


def main():
    """Main entry point."""
    levels_dir = os.path.join(project_root, "levels")
    all_spawn_path = os.path.join(project_root, "gamedata", "spawns", "all.spawn")
    level_id = None

    # Get file path from command line or show selection menu
    if len(sys.argv) > 1:
        level_file = sys.argv[1]
        # If path doesn't exist, try resolving from project root
        if not os.path.exists(level_file):
            level_file = os.path.join(project_root, sys.argv[1])
        # Try to determine level_id from the all.spawn game graph
        try:
            gg = GameGraphParser.from_all_spawn(Path(all_spawn_path))
            levels = gg.get_levels()
            # Match by level name extracted from path (e.g. levels/l01_escape/level.ai -> l01_escape)
            level_dir_name = os.path.basename(os.path.dirname(os.path.abspath(level_file)))
            for lid, level in levels.items():
                if level.name == level_dir_name:
                    level_id = lid
                    break
        except Exception:
            pass
    else:
        level_file, level_id = select_level(all_spawn_path, levels_dir)

    if not os.path.exists(level_file):
        print(f"Error: File not found: {level_file}")
        print("Usage: python visualiser/run_visualiser.py <path/to/level.ai>")
        sys.exit(1)

    # GUID validation
    if level_id is not None:
        _validate_guid(level_file, all_spawn_path, level_id)

    print()
    print("=" * 60)
    print("LEVEL VERTEX GRAPH INSPECTOR")
    print("=" * 60)
    print(f"Loading: {level_file}")
    print("Starting at vertex 0")
    print("Ctrl+Click to jump to nearest vertex or spawn object")
    print("Use arrow keys to navigate via vertex connections")
    print("Press Space to focus camera on selected vertex")
    print("=" * 60)

    import open3d.visualization.gui as gui
    from ui import NodeInspectorApp

    # Initialize Open3D GUI
    gui.Application.instance.initialize()

    # Create and run application
    app = NodeInspectorApp(level_file, level_id=level_id, all_spawn_path=all_spawn_path)
    app.run()


if __name__ == "__main__":
    main()
