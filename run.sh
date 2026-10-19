#!/usr/bin/env bash

set -Eeuo pipefail

if [[ ${1:-} == --dry-run ]]; then
    DRY_RUN=1
    shift
else
    DRY_RUN=0
fi
if [[ $# -ne 2 ]]; then
    echo "usage: $0 [--dry-run] config.yaml run-name" >&2
    exit 2
fi
CONFIG=$1
RUN_NAME=$2
[[ $RUN_NAME =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
    echo "run-name must use only letters, digits, underscore, dot, or hyphen" >&2
    exit 2
}

WDIR=/root/mpitest
SCRIPT=dist_train.py
FILES=(
    timetree.py protocol.py transport.py lifecycle.py ps.py
    runtime.py training.py models.py data.py topology.py
)

[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "missing $SCRIPT" >&2; exit 1; }
[[ -f launcher.py ]] || { echo "missing launcher.py" >&2; exit 1; }
for file in "${FILES[@]}"; do [[ -f "$file" ]] || { echo "missing $file" >&2; exit 1; }; done

# Validate the complete topology before any SSH/SCP operation. The regular file
# preserves Python's exit status (process substitution would mask a failure).
PLAN_FILE=$(mktemp)
trap 'rm -f "$PLAN_FILE"' EXIT
python3 launcher.py "$CONFIG" > "$PLAN_FILE"

declare -A ALL_HOSTS=()
declare -a EDGE_IDS=() EDGE_HOSTS=() WORKER_IDS=() WORKER_HOSTS=()
ROOT_ID= ROOT_HOST= EVAL_ID= EVAL_HOST=
while IFS=$'\t' read -r host node_id role; do
    ALL_HOSTS["$host"]=1
    case "$role" in
        root_ps) ROOT_ID=$node_id; ROOT_HOST=$host ;;
        edge_ps) EDGE_IDS+=("$node_id"); EDGE_HOSTS+=("$host") ;;
        eval) EVAL_ID=$node_id; EVAL_HOST=$host ;;
        worker) WORKER_IDS+=("$node_id"); WORKER_HOSTS+=("$host") ;;
        *) echo "unknown role: $role" >&2; exit 1 ;;
    esac
done < "$PLAN_FILE"

if (( DRY_RUN )); then
    printf 'Deploy once to host: %s\n' "${!ALL_HOSTS[@]}"
    while IFS=$'\t' read -r host node_id role; do
        printf 'Launch %s on %s: venv/bin/python -u %s --config config.yaml --node-id %s\n' \
            "$role" "$host" "$SCRIPT" "$node_id"
    done < "$PLAN_FILE"
    exit 0
fi

mkdir -p logs
echo "Deploying files..."
for host in "${!ALL_HOSTS[@]}"; do
    (
        ssh "$host" "test -d '$WDIR'" ||
            { echo "error: $WDIR does not exist on $host"; exit 1; }
        scp "$SCRIPT" "$host:$WDIR/"
        scp "${FILES[@]}" "$host:$WDIR/"
        scp "$CONFIG" "$host:$WDIR/config.yaml"
    ) &
done
wait

LOG_DIR="$WDIR/logs"
SSH_FLAGS=(-o StrictHostKeyChecking=no)

start_node() {
    local host=$1 node_id=$2
    local logfile="$LOG_DIR/${RUN_NAME}_${node_id}.log"
    ssh "${SSH_FLAGS[@]}" "$host" \
        "mkdir -p '$LOG_DIR'; cd '$WDIR' && venv/bin/python -u '$SCRIPT' --config config.yaml --node-id '$node_id' > '$logfile' 2>&1 &"
}

echo "Starting PS nodes..."
start_node "$ROOT_HOST" "$ROOT_ID"
for i in "${!EDGE_IDS[@]}"; do start_node "${EDGE_HOSTS[i]}" "${EDGE_IDS[i]}"; done

sleep 1
echo "Starting evaluator and workers..."
start_node "$EVAL_HOST" "$EVAL_ID"
for i in "${!WORKER_IDS[@]}"; do start_node "${WORKER_HOSTS[i]}" "${WORKER_IDS[i]}"; done

echo "All processes started. Collecting root logs to logs/${RUN_NAME}_all.log"
ssh "${SSH_FLAGS[@]}" "$ROOT_HOST" "tail -f '$LOG_DIR/${RUN_NAME}_${ROOT_ID}.log'" 2>/dev/null | \
    tee "logs/${RUN_NAME}_all.log" \
        >(grep --line-buffered "EVAL"    > "logs/${RUN_NAME}_eval.log") \
        >(grep --line-buffered "METRICS" > "logs/${RUN_NAME}_metrics.log") &

wait
