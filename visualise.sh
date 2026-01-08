#!/bin/bash
# Launch the node graph visualiser
#
# Usage:
#   ./visualise.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: python not found" >&2
    exit 1
fi

# Check for required dependencies
if ! $PYTHON -c "import open3d" &> /dev/null; then
    echo "Error: open3d is not installed" >&2
    echo "Install it with: $PYTHON -m pip install open3d" >&2
    exit 1
fi

cd "$SCRIPT_DIR/visualiser" || exit 1

$PYTHON run_visualiser.py "$@"
