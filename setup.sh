#!/usr/bin/env bash
# ============================================================================
#  GodhomeRL -- one-command setup for Linux and macOS
#
#  Usage:
#      ./setup.sh                                  build into the current env
#      ./setup.sh --venv .venv                      build into a new venv
#      ./setup.sh --install ~/games/godhome-...    explicit game location
#
#  After it finishes:
#      python -m godhome_rl doctor
#      python -m godhome_rl train --name godhome --bosses all --n-envs 3
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# 0. Locate a Python interpreter
# ---------------------------------------------------------------------------
PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done
if [ -z "$PY" ]; then
    echo "[setup] Python 3.10+ is required but was not found." >&2
    echo "[setup] Install it, or activate an existing virtualenv and re-run." >&2
    exit 1
fi
echo "[setup] using $($PY --version) at $(command -v "$PY")"

# ---------------------------------------------------------------------------
# 1. Arguments
# ---------------------------------------------------------------------------
VENV=""
INSTALL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --venv)    VENV="$2"; shift 2 ;;
        --install) INSTALL="$2"; shift 2 ;;
        *) echo "[setup] ignoring unknown argument '$1'"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# 2. Optional virtual environment
# ---------------------------------------------------------------------------
if [ -n "$VENV" ]; then
    if [ ! -x "$VENV/bin/python" ]; then
        echo "[setup] creating a virtual environment at $VENV"
        "$PY" -m venv "$VENV"
    fi
    PY="$VENV/bin/python"
fi

# ---------------------------------------------------------------------------
# 3. Dependencies
# ---------------------------------------------------------------------------
echo "[setup] installing dependencies"
"$PY" -m pip install --upgrade pip --quiet
"$PY" -m pip install -e . --quiet

# ---------------------------------------------------------------------------
# 4. Locate the Godhome install and build
# ---------------------------------------------------------------------------
BUILD_ARGS=()
if [ -n "$INSTALL" ]; then
    BUILD_ARGS=(--install "$INSTALL")
elif [ -n "${GODHOME_RL_INSTALL:-}" ]; then
    BUILD_ARGS=(--install "$GODHOME_RL_INSTALL")
fi

echo "[setup] building the patched game (first run unpacks ~4000 assets)"
"$PY" -m godhome_rl build "${BUILD_ARGS[@]}"

# ---------------------------------------------------------------------------
# 5. On Linux the game needs an OpenGL context even when headless; install a
#    software rasteriser if nothing else is available.
# ---------------------------------------------------------------------------
if [ "$(uname -s)" = "Linux" ]; then
    if ! ldconfig -p 2>/dev/null | grep -q 'libGL.so.1'; then
        echo "[setup] installing libgl1-mesa-dri (software OpenGL)"
        if command -v sudo >/dev/null 2>&1; then
            sudo apt-get update -qq && sudo apt-get install -y -qq libgl1-mesa-dri || true
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 6. Verify
# ---------------------------------------------------------------------------
"$PY" -m godhome_rl doctor

cat <<'EOF'

[setup] done.
EOF
if [ -n "$VENV" ]; then
    echo "[setup] activate with:  source $VENV/bin/activate"
fi
echo "[setup] then run:  python -m godhome_rl train --name godhome --bosses all --n-envs 3"
echo "[setup]           python -m godhome_rl play  --model runs/godhome/checkpoints/latest.pt"