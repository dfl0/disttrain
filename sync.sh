#!/usr/bin/env bash
set -euo pipefail
if (($# != 2)); then
    echo "Usage: $0 FILE_LIST HOST:REMOTE_DIRECTORY" >&2
    exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

files=()
while IFS= read -r file || [[ -n "$file" ]]; do
    [[ -n "$file" ]] && files+=("$file")
done < "$1"

((${#files[@]})) || { echo "$1 is empty" >&2; exit 1; }
for file in "${files[@]}"; do
    [[ -e "$file" ]] || { echo "Missing file: $file" >&2; exit 1; }
done

scp -r "${files[@]}" "$2"
