#!/usr/bin/env bash
# copy runtime files to one device
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if (($# == 0)); then
    echo "Usage: $0 HOST --workdir PATH [--setup] [--python python3] [--force] [--dry-run]" >&2
    exit 2
fi
python="${LAUNCHER_PYTHON:-$script_dir/venv/bin/python}"
if [[ ! -x "$python" ]] && [[ -z "${LAUNCHER_PYTHON:-}" ]]; then
    python=python3
fi
if [[ $1 == --help || $1 == -h ]]; then
    exec "$python" "$script_dir/setup_devices.py" --help
fi
host=$1
shift
exec "$python" "$script_dir/setup_devices.py" --host "$host" "$@"
