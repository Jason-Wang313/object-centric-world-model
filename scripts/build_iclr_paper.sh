#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="paper/build"
TEX="paper/object_binding_tail_audit_iclr.tex"
PDF="paper/object_binding_tail_audit_iclr.pdf"

mkdir -p "$BUILD_DIR"

if command -v xelatex >/dev/null 2>&1; then
  LATEX=xelatex
elif command -v xelatex.exe >/dev/null 2>&1; then
  LATEX=xelatex.exe
elif [[ -x /mnt/c/Users/wangz/AppData/Local/Programs/MiKTeX/miktex/bin/x64/xelatex.exe ]]; then
  LATEX=/mnt/c/Users/wangz/AppData/Local/Programs/MiKTeX/miktex/bin/x64/xelatex.exe
else
  echo "xelatex not found" >&2
  exit 1
fi

"$LATEX" -interaction=nonstopmode -halt-on-error -output-directory "$BUILD_DIR" "$TEX"
"$LATEX" -interaction=nonstopmode -halt-on-error -output-directory "$BUILD_DIR" "$TEX"

cp "$BUILD_DIR/object_binding_tail_audit_iclr.pdf" "$PDF"
echo "Wrote $PDF"
