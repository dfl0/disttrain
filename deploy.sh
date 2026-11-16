#!/usr/bin/env bash
# copy runtime files to one device
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./deploy.sh HOST --workdir /absolute/remote/path [--setup] [--python python3] [--dry-run]

Copies all selected runtime files in one SCP transfer over SSH.
--setup    Create venv if needed and install requirements_node.txt remotely.
--python   Remote Python used to create venv; default python3.
--dry-run  Print files and commands only; no SSH connection or remote changes.

HOST accepts an SSH alias or user@host. Paths must contain only letters,
digits, underscore, dot, slash or hyphen. Existing files are updated, but
unlisted files, datasets, logs and environments are never deleted.
EOF
}

host=""
workdir=""
remote_python=python3
setup=false
dry_run=false
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --workdir|--python)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            if [[ $1 == --workdir ]]; then workdir=$2; else remote_python=$2; fi
            shift 2 ;;
        --setup) setup=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            if [[ -n $host ]]; then echo "Expected one host" >&2; exit 2; fi
            host=$1; shift ;;
    esac
done

# restrict arguments passed to the remote shell
if [[ ! $host =~ ^([A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Provide an SSH host alias or user@host" >&2; exit 2
fi
if [[ ! $workdir =~ ^/[A-Za-z0-9_./-]+$ || $workdir == / ]]; then
    echo "--workdir requires an absolute project path without spaces or shell characters" >&2; exit 2
fi
if [[ ! $remote_python =~ ^[A-Za-z0-9_./][A-Za-z0-9_./-]*$ ]]; then
    echo "--python requires an executable name or path without spaces or shell characters" >&2; exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir"
files=(
    dist_train.py launcher.py runtime.py topology.py lifecycle.py
    training.py ps.py async_ps.py protocol.py transport.py
    models.py data.py metrics.py requirements_node.txt
)
for file in "${files[@]}"; do
    [[ -f $file ]] || { echo "Missing runtime file: $file" >&2; exit 1; }
done
command -v ssh >/dev/null
command -v scp >/dev/null

run() {
    printf "%q " "$@"
    printf "\n"
    if ! $dry_run; then "$@"; fi
}

ssh_options=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
run ssh -T "${ssh_options[@]}" "$host" "mkdir -p '$workdir'"
run scp "${ssh_options[@]}" -- "${files[@]}" "$host:$workdir/"

if $setup; then
    run ssh -T "${ssh_options[@]}" "$host" \
        "cd '$workdir' && { test -x venv/bin/python || '$remote_python' -m venv venv; } && venv/bin/python -m pip install -r requirements_node.txt"
fi

if $dry_run; then
    echo "Preview only; no remote checks or changes."
else
    echo "Runtime copied to $host:$workdir. Use this path as the launcher's --workdir."
fi
