#!/bin/bash
# Launch the node graph visualiser
#
# Usage:
#   ./visualise.sh

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
if ! $PYTHON -c "import open3d" &> /dev/null; then
    echo "Error: open3d is not installed" >&2
    echo "Install it with: $PYTHON -m pip install open3d" >&2
    read -p "Press Enter to exit..."
    exit 1
fi

cd "$SCRIPT_DIR/visualiser" || { echo "Error: failed to change to visualiser directory" >&2; read -p "Press Enter to exit..."; exit 1; }

$PYTHON run_visualiser.py "$@"
