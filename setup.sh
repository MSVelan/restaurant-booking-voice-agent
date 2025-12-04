if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Run: source ./setup.sh  (or . ./setup.sh) \
to activate the virtualenv." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  echo "Running: uv sync"
  uv sync || { echo "uv sync failed"; return 1; }
else
  echo "uv not found in PATH. Please install uv first."
fi

VENV_ACTIVATE_LINUX=".venv/bin/activate"
VENV_ACTIVATE_WINDOWS=".venv/Scripts/activate"

echo "Activating .venv..."
if [ -f "$VENV_ACTIVATE_LINUX" ]; then
    source "$VENV_ACTIVATE_LINUX"
elif [ -f "$VENV_ACTIVATE_WINDOWS" ]; then
    source "$VENV_ACTIVATE_WINDOWS"
else
  echo ".venv/bin/activate not found. uv sync should have created .venv" >&2
  return 1
fi

echo "Installing editable package with: uv pip install -e ."
uv pip install -e . || { echo "uv pip install -e . failed"; return 1; }