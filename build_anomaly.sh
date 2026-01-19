#!/bin/bash
# Build all.spawn - main game graph compilation script
#
# Usage:
#   ./build_anomaly.sh              # Normal build
#   ./build_anomaly.sh --force      # Force rebuild all cross tables

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find python - prefer virtual environment if available
if [[ -f "$SCRIPT_DIR/venv/Scripts/python.exe" ]]; then
    PYTHON="$SCRIPT_DIR/venv/Scripts/python.exe"
elif [[ -f "$SCRIPT_DIR/.venv/Scripts/python.exe" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/Scripts/python.exe"
elif [[ -f "$SCRIPT_DIR/venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
elif [[ -f "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: python not found" >&2
    read -p "Press Enter to exit..."
    exit 1
fi

# Check for required dependencies
if ! $PYTHON -c "import numpy" &> /dev/null; then
    echo "Error: numpy is not installed" >&2
    echo "Install it with: $PYTHON -m pip install numpy" >&2
    read -p "Press Enter to exit..."
    exit 1
fi

cd "$SCRIPT_DIR/compiler" || { echo "Error: failed to change to compiler directory" >&2; read -p "Press Enter to exit..."; exit 1; }

$PYTHON build_all_spawn.py \
    --config ../levels.ini \
    --output ../gamedata/spawns/all.spawn \
    --blacklist ../spawn_blacklist.ini \
    --basemod anomaly
    "$@"
read -p "Press Enter to exit..."
    exit 1