#!/bin/bash
# Build the two native dependencies the benchmark needs:
#   1. libbg_engine.so  -- the C backgammon engine behind `bg_fast` (our move-gen)
#   2. gnubg-nn          -- GNU Backgammon neural eval, a published PyPI wheel
#
# No hard-coded paths: everything is resolved relative to this script.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

echo "==> Building libbg_engine.so (bg_fast) ..."
( cd "$REPO/c_engine" && bash build_unix.sh )

echo "==> Ensuring gnubg-nn==1.1.0a8 is installed ..."
if python3 -c "import gnubg_nn" 2>/dev/null; then
    echo "    gnubg_nn already importable (v$(python3 -c 'from importlib.metadata import version; print(version("gnubg-nn"))'))"
else
    # Published wheel on PyPI; pinned to the version the paper numbers used.
    pip install "gnubg-nn==1.1.0a8"
fi

echo "==> Sanity check ..."
cd "$REPO"
python3 -c "
import sys; sys.path.insert(0, 'c_engine')
import bg_fast, gnubg_nn
from td_agent import TDAgent
from agents import GnubgNNAgent
print('    OK: bg_fast + gnubg_nn + engine import cleanly')
"
echo "Build complete. Run ./run.sh to reproduce the numbers."
