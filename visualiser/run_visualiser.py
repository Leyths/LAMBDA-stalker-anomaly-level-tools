#!/usr/bin/env python3
"""
Node Graph Inspector - Main Entry Point

A tool for inspecting navigation mesh node graphs (read-only).
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
compiler_levels = os.path.join(project_root, "compiler/levels")
compiler_root = os.path.join(project_root, "compiler")

sys.path.insert(0, compiler_levels)
sys.path.insert(0, compiler_root)
from levels_config import LevelsConfig

# Remove compiler's utils from module cache so visualiser's utils can be loaded
if 'utils' in sys.modules:
    del sys.modules['utils']

# Now add visualiser directory to Python path for UI imports
sys.path.insert(0, script_dir)

# Now we can remove compiler_path if needed (though keeping it shouldn't hurt)
# sys.path.remove(compiler_path)
def select_level():
    """Show level selection menu and return the selected level.ai path and level_id."""
    config_path = os.path.join(project_root, "levels.ini")
    try:
        config = LevelsConfig(config_path)
    except Exception as e:
        print(f"Error loading levels.ini: {e}")
        sys.exit(1)

    print("=" * 60)
    print("LEVEL SELECTION")
    print("=" * 60)
    print()

    # Build list of levels with existing level.ai files
    available_levels = []
    for level in config.levels:
        # Convert path from compiler-relative to project-relative
        level_path = level.path.replace("../", "")
        ai_path = os.path.join(project_root, level_path, "level.ai")
        if os.path.exists(ai_path):
            available_levels.append((level, ai_path))

    if not available_levels:
        print("No levels with level.ai files found!")
        sys.exit(1)

    # Display menu
    for i, (level, _) in enumerate(available_levels, 1):
        print(f"  {i:2d}. {level.name}")

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
            if 0 <= idx < len(available_levels):
                level, ai_path = available_levels[idx]
                return ai_path, level.id
            else:
                print(f"Please enter a number between 1 and {len(available_levels)}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)


def main():
    """Main entry point."""
    level_id = None

    # Get file path from command line or show selection menu
    if len(sys.argv) > 1:
        level_file = sys.argv[1]
        # If path doesn't exist, try resolving from project root
        if not os.path.exists(level_file):
            level_file = os.path.join(project_root, sys.argv[1])
        # Try to determine level_id from path for command-line usage
        # Load config to find matching level
        try:
            config = LevelsConfig(os.path.join(project_root, "levels.ini"))
            for level in config.levels:
                level_path = level.path.replace("../", "")
                ai_path = os.path.join(project_root, level_path, "level.ai")
                if os.path.abspath(ai_path) == os.path.abspath(level_file):
                    level_id = level.id
                    break
        except Exception:
            pass
    else:
        level_file, level_id = select_level()

    if not os.path.exists(level_file):
        print(f"Error: File not found: {level_file}")
        print("Usage: ./visualise.sh <path/to/level.ai>")
        sys.exit(1)

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
    all_spawn_path = os.path.join(project_root, "gamedata", "spawns", "all.spawn")
    app = NodeInspectorApp(level_file, level_id=level_id, all_spawn_path=all_spawn_path)
    app.run()


if __name__ == "__main__":
    main()
