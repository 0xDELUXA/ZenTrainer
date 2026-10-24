#!/usr/bin/env sh
# ZenTrainer launcher for Linux (on Windows: start-windows.bat).
#
# It does three things, then hands over to zt (cli/: one Python program for both systems):
#   1. checks this is Linux
#   2. makes sure uv is installed (uv is a Python manager: it downloads the right Python itself)
#   3. runs zt with that Python: first-run setup, update check, the menu
#
# Arguments pass through to zt, e.g. `./start-linux.sh start --open`. Only git is
# needed beforehand. Nothing is installed system-wide: uv goes into your home folder.
set -e
cd "$(dirname "$0")"

# ---- settings -----------------------------------------------------------------------------------
UV_INSTALLER="https://astral.sh/uv/install.sh"   # uv's official installer (https://docs.astral.sh/uv/)
UV_DIR="$HOME/.local/bin"                         # where that installer puts uv
CLI="cli"                                         # zt, the Python program that does the rest
# the Python version comfy.lock pins (the managed ComfyUI uses the same one)
PYTHON_VERSION=$(sed -n 's/.*"python": *"\([^"]*\)".*/\1/p' comfy.lock)
PYTHON_VERSION=${PYTHON_VERSION:-3.12}

# ---- 1. this is Linux ---------------------------------------------------------------------------
case "$(uname -s)" in
  Linux) ;;
  MINGW*|MSYS*|CYGWIN*) echo "This is the Linux launcher: on Windows, run start-windows.bat."; exit 1 ;;
  Darwin) echo "macOS isn't supported yet: ZenTrainer's pinned PyTorch builds are for NVIDIA (CUDA), AMD (ROCm) and CPU."; exit 1 ;;
  *) echo "This is the Linux launcher, and this system is $(uname -s): continuing, but it's untested here." ;;
esac

# ---- 2. uv --------------------------------------------------------------------------------------
PATH="$UV_DIR:$PATH"
export PATH
if ! command -v uv >/dev/null 2>&1; then
  echo "uv isn't installed. ZenTrainer uses it to get Python $PYTHON_VERSION and its packages."
  echo "  installer: $UV_INSTALLER"
  echo "  installs to: $UV_DIR"
  printf "Install uv now? [Y/n] "
  read -r answer || answer="n"
  case "$answer" in [nN]*) echo "Not installed. Install uv yourself, then run this again."; exit 1 ;; esac
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "$UV_INSTALLER" | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$UV_INSTALLER" | sh
  else
    echo "Needs curl or wget to download uv."; exit 1
  fi
fi

# ---- 3. hand over to zt -------------------------------------------------------------------------
# uv downloads Python $PYTHON_VERSION and zt's packages (cli/uv.lock) the first time, into cli/.venv,
# then reuses them. --frozen: use the lock as it is, never rewrite it.
exec uv run --quiet --frozen --project "$CLI" --python "$PYTHON_VERSION" zt "$@"
