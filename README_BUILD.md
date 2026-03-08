# Building from Source

Instructions for developers building LAMBDA from source or contributing to the project.

## Prerequisites

- Python 3.11+
- `numpy` (required)
- `open3d` (required for visualiser)

## Setup

```bash
git clone <repo>
cd LAMBDA-stalker-anomaly-level-tools
python3 -m venv venv
source venv/bin/activate
pip install numpy open3d
```

## Running the GUI

```bash
python lambda.py
```

The GUI provides the same functionality as `LAMBDA.exe` — build tab, visualiser tab, and all configuration options.

## Running from the command line

### Via lambda.py

The `--build` flag launches the GUI and automatically starts a build:

```bash
# Launch GUI and auto-start build
python lambda.py --build --basemod anomaly

# With directory overrides
python lambda.py --build \
    --levels-dir /path/to/levels \
    --output-dir /path/to/gamedata \
    --basemod gamma

# Deploy only (skip spawn rebuild, copy mod files only)
python lambda.py --build --deploy-only

# Level file overrides (partial directory, falls back to default levels)
python lambda.py --build --levels-override-dir /path/to/override/levels
```

### Via build_all_spawn.py (direct)

For headless builds without the GUI:

```bash
cd compiler

# Default build
python build_all_spawn.py --project-root .. --basemod anomaly

# Override directories
python build_all_spawn.py \
    --project-root .. \
    --levels-dir /path/to/levels \
    --output-dir /path/to/gamedata \
    --basemod gamma

# Force rebuild all cross tables
python build_all_spawn.py --project-root .. --force

# Deploy mod scripts only (skip spawn rebuild)
python build_all_spawn.py --project-root .. --deploy-only --basemod anomaly
```

### CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--project-root` | `..` | Project root directory |
| `--levels-dir` | `<project>/levels` | Override levels directory |
| `--levels-override-dir` | None | Partial level file overrides (per-file fallback to levels-dir) |
| `--output-dir` | `<project>/gamedata` | Override output directory |
| `--force` | off | Force rebuild all cross tables |
| `--deploy-only` | off | Skip spawn build, copy mod files only |
| `--basemod` | `anomaly` | Target base mod (`anomaly`, `cultured`, `gamma`) |

## Building the Windows EXE

Run `build_exe.bat` on a Windows machine with Python 3.11+ in PATH. It creates a `dist/` folder containing the EXE and all runtime files, ready to zip and distribute.

The script:
1. Creates a build virtual environment
2. Installs PyInstaller, numpy, and open3d
3. Runs PyInstaller with `lambda.spec`
4. Copies config files, anomaly data, levels, mods, and docs into `dist/`

## Project Structure

```
├── lambda.py                # GUI / CLI entry point
├── lambda.spec              # PyInstaller spec file
├── build_exe.bat            # Builds LAMBDA.exe from source
├── levels.ini               # Level definitions
├── anomaly.ini              # Anomaly mod config
├── cultured.ini             # Cultured mod config
├── gamma.ini                # GAMMA mod config
├── spawn_blacklist.ini      # Entity exclusion patterns
├── level_changers.ini       # Cross-level teleport config
├── anomaly/                 # Pre-extracted spawn/patrol/edge data
├── levels/                  # Level data (level.ai, level.spawn, level.game)
├── mods/                    # Mod overlay files
├── gamedata/                # Build output
├── .tmp/                    # Build cache (cross tables, staged levels)
├── compiler/                # Build pipeline
│   ├── build_all_spawn.py   #   Master orchestrator (GameGraphBuilder)
│   ├── project_paths.py     #   Path resolution
│   ├── game_graph_merger.py #   Graph merging
│   ├── config/              #   Mod config, copier, tag rewriter
│   ├── converters/          #   Spawn data converters/exporters
│   ├── crosstables/         #   Cross table building
│   ├── extraction/          #   Spawn entity extraction
│   ├── generation/          #   Death point generation
│   ├── graph/               #   GameGraph data structure
│   ├── levels/              #   Level configuration parser
│   ├── patrols/             #   Patrol path handling
│   ├── parsers/             #   Binary format parsers
│   ├── remapping/           #   GVID remapping
│   ├── serialization/       #   Binary output (all.spawn writer)
│   ├── spawn_graph/         #   Spawn graph building
│   └── utils/               #   Logging, shared utilities
├── visualiser/              # 3D level inspector
│   ├── run_visualiser.py    #   Entry point
│   ├── core/                #   Data loading
│   ├── ui/                  #   Open3D rendering
│   └── utils/               #   Helpers
└── docs/                    # Documentation
```
