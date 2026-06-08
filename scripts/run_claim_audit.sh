#!/usr/bin/env bash
set -euo pipefail

pick_python() {
  for candidate in python.exe python python3 py.exe py; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import json" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(pick_python)" || {
  echo "No Python executable found on PATH" >&2
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY_ROOT="$ROOT_DIR"
PY_SRC="$ROOT_DIR/src"
if [[ "$PYTHON_BIN" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then
  PY_ROOT="$(wslpath -w "$ROOT_DIR")"
  PY_SRC="$(wslpath -w "$ROOT_DIR/src")"
elif command -v cygpath >/dev/null 2>&1; then
  PY_ROOT="$(cygpath -w "$ROOT_DIR")"
  PY_SRC="$(cygpath -w "$ROOT_DIR/src")"
fi
export PYTHONPATH="$PY_SRC"

"$PYTHON_BIN" src/object_centric_best_of_n/audit.py --root "$PY_ROOT"
