#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Error: Python not found. Please install python3 first." >&2
  exit 1
fi

echo "Using Python: $PYTHON_BIN"

if [[ ! -d "$ROOT_DIR/.venv" ]]; then
  echo "Creating virtual environment at .venv ..."
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
else
  echo "Virtual environment already exists: .venv"
fi

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
VENV_PIP="$ROOT_DIR/.venv/bin/pip"

echo "Upgrading pip ..."
"$VENV_PYTHON" -m pip install --upgrade pip

echo "Installing requirements ..."
"$VENV_PIP" install -r "$ROOT_DIR/requirements.txt"

echo
echo "Setup complete."
echo "Next step:"
echo "  ./run.sh"
