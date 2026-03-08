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

```bash
cd compiler
python build_all_spawn.py \
    --config ../levels.ini \
    --levels-dir /path/to/your/levels \
    --output /path/to/gamedata/spawns/all.spawn \
    --basemod anomaly

# Force rebuild all cross tables
python build_all_spawn.py --config ../levels.ini --force
```

## Building the Windows EXE

Run `build_exe.bat` on a Windows machine with Python 3.11+ in PATH. It creates a `dist/` folder containing the EXE and all runtime files, ready to zip and distribute.

The script:
1. Creates a build virtual environment
2. Installs PyInstaller, numpy, and open3d
3. Runs PyInstaller with `lambda.spec`
4. Copies config files, anomaly data, levels, mods, and docs into `dist/`

## Project Structure

```
├── LAMBDA.exe               # Windows GUI (standalone, no Python needed)
├── lambda.py                # GUI source (macOS/Linux)
├── lambda.spec              # PyInstaller spec file
├── build_exe.bat            # Builds LAMBDA.exe from source
├── levels.ini               # Level definitions
├── anomaly.ini              # Anomaly mod configuration
├── cultured.ini             # Cultured mod configuration
├── gamma.ini                # GAMMA mod configuration
├── spawn_blacklist.ini      # Entity exclusion patterns
├── level_changers.ini       # Cross-level teleport config
├── anomaly/                 # Pre-extracted spawn/patrol/edge data
├── compiler/                # Build pipeline (Python)
├── levels/                  # Level data directories
├── mods/                    # Mod overlay files
├── gamedata/                # Build output
├── .tmp/                    # Build cache
├── visualiser/              # 3D inspector tool
└── docs/                    # Documentation
```
