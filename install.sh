#!/bin/bash
# Script to run the goan installation from the repo root.
# This will create a .venv_goan in the 'install' subdirectory.

# Ensure Python 3 is in PATH
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not found. Please install Python 3 and ensure it's in your PATH." >&2
    exit 1
fi

# Determine the directory of the current script (repo root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
INSTALL_DIR="$SCRIPT_DIR/install"
INSTALL_SCRIPT="$INSTALL_DIR/install.py"

if [ ! -d "$INSTALL_DIR" ]; then
    echo "Error: The 'install' directory not found at '$INSTALL_DIR'." >&2
    exit 1
fi
if [ ! -f "$INSTALL_SCRIPT" ]; then
    echo "Error: 'install.py' not found at '$INSTALL_SCRIPT'." >&2
    exit 1
fi

# Pass arguments to the Python script
python3 "$INSTALL_SCRIPT" "$@"
