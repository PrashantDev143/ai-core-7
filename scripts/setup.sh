#!/usr/bin/env bash
# Creates backend/.venv and installs dependencies.
# torch comes from the CPU index first; pip's default wheel is the ~2.5GB CUDA
# build, which is dead weight without an Nvidia GPU.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$root/backend/.venv"

if [ ! -d "$venv" ]; then
    python3 -c 'import sys; assert sys.version_info >= (3,11)' \
        || { echo "Python 3.11+ required"; exit 1; }
    python3 -m venv "$venv"
fi

"$venv/bin/python" -m pip install --upgrade pip setuptools wheel --quiet
"$venv/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$venv/bin/python" -m pip install -e "$root/backend[dev]"

echo
echo "Done. Activate with:"
echo "  source backend/.venv/bin/activate"
