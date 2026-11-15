#!/usr/bin/env bash
set -euo pipefail
# prefer the local venv unless LAUNCHER_PYTHON is set
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n ${LAUNCHER_PYTHON:-} ]]; then
    PYTHON=$LAUNCHER_PYTHON
elif [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON=python3
fi
exec "$PYTHON" "$SCRIPT_DIR/launcher.py" "$@"
