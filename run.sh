#!/usr/bin/env bash

set -Eeuo pipefail

HOSTFILE=hosts
WDIR=/root/mpitest
SCRIPT=dist_train.py
FILES=(
    timetree.py
)
CONFIG=$1

[[ -f "$HOSTFILE" ]] || { echo "missing $HOSTFILE"; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG"; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "missing $SCRIPT"; exit 1; }
for file in "${FILES[@]}"; do [[ -f "$file" ]] || { echo "missing $file"; exit 1; }; done

NUM_HOSTS=$(grep '^\s*\-' $CONFIG | wc -l)
TOTAL_HOSTS=$(wc -l < "$HOSTFILE")

if ((NUM_HOSTS + 1 > TOTAL_HOSTS)); then
    echo "error: hierarchy config specifies more hosts than are available"
    exit 1
fi

if ((NUM_HOSTS + 2 > TOTAL_HOSTS)); then
    echo "error: extra host required as evaluator"
    exit 1
fi

max_hosts=$((NUM_HOSTS + 2))
host_arr=()
i=0

echo "Initializing hosts"
while IFS= read -r host; do
    host_arr+=("$host")
    (
        ssh "$host" "test -d '$WDIR'" ||
        { echo "error: $WDIR does not exist on $host, make sure device is set up"; exit 1; }

        scp "$SCRIPT" "$host:$WDIR/"
        scp "${FILES[@]}" "$host:$WDIR/"
        scp "$CONFIG" "$host:$WDIR/config.yaml"
    ) &
    ((++i >= max_hosts)) && break
done < "$HOSTFILE"

wait -n || { echo "host setup failed, aborting"; kill 0; exit 1; }
wait

hosts=$(IFS=,; echo "${host_arr[*]}")
stdbuf -oL -eL \
mpiexec --allow-run-as-root \
        --host $hosts \
        --map-by ppr:1:node \
        --wdir "$WDIR" \
        venv/bin/python -u "$SCRIPT" 2>&1 | \
tee "logs/$2_all.log" \
    >(grep --line-buffered "EVAL" | grep --line-buffered -v "METRICS" > "logs/$2_eval.log") \
    >(grep --line-buffered "METRICS" > "logs/$2_metrics.log")
