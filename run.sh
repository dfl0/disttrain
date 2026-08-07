#!/usr/bin/env bash

set -Eeuo pipefail

CONFIG=$1
RUN_NAME=$2

WDIR=/root/mpitest
SCRIPT=dist_train.py
FILES=(
    timetree.py
    comm.py
    ps.py
)

[[ -f "$CONFIG" ]] || { echo "missing $CONFIG"; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "missing $SCRIPT"; exit 1; }
for file in "${FILES[@]}"; do [[ -f "$file" ]] || { echo "missing $file"; exit 1; }; done

# Parse all node addresses from the config using Python
read -r ROOT_ADDR EVAL_HOST < <(python3 - "$CONFIG" <<'EOF'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
root = cfg.get("bind_addr", "127.0.0.1")
evh  = cfg.get("eval", {}).get("host", "127.0.0.1")
print(root, evh)
EOF
)

mapfile -t EDGE_ADDRS < <(python3 - "$CONFIG" <<'EOF'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
for e in cfg.get("edge_servers", []):
    print(e["bind_addr"])
EOF
)

mapfile -t WORKER_HOSTS < <(python3 - "$CONFIG" <<'EOF'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
for e in cfg.get("edge_servers", []):
    for w in e.get("workers", []):
        print(w["hostname"])
for w in cfg.get("workers", []):
    print(w["hostname"])
EOF
)

# Collect all unique hosts for file deployment
declare -A ALL_HOSTS
ALL_HOSTS[$ROOT_ADDR]=1
ALL_HOSTS[$EVAL_HOST]=1
for h in "${EDGE_ADDRS[@]}"; do ALL_HOSTS[$h]=1; done
for h in "${WORKER_HOSTS[@]}"; do ALL_HOSTS[$h]=1; done

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

PYEXE="venv/bin/python -u $SCRIPT"
LOG_DIR="$WDIR/logs"
SSH_FLAGS="-o StrictHostKeyChecking=no"

echo "Starting PS nodes..."

# Root PS binds first
ssh $SSH_FLAGS "$ROOT_ADDR" "mkdir -p $LOG_DIR; cd $WDIR && $PYEXE > $LOG_DIR/${RUN_NAME}_root.log 2>&1 &"

# Edge PS nodes
for edge in "${EDGE_ADDRS[@]}"; do
    ssh $SSH_FLAGS "$edge" "mkdir -p $LOG_DIR; cd $WDIR && $PYEXE > $LOG_DIR/${RUN_NAME}_edge_${edge}.log 2>&1 &"
done

# Give PS nodes a moment to bind their ports
sleep 1

echo "Starting evaluator and workers..."

ssh $SSH_FLAGS "$EVAL_HOST" "mkdir -p $LOG_DIR; cd $WDIR && $PYEXE > $LOG_DIR/${RUN_NAME}_eval.log 2>&1 &"

for whost in "${WORKER_HOSTS[@]}"; do
    ssh $SSH_FLAGS "$whost" "mkdir -p $LOG_DIR; cd $WDIR && $PYEXE > $LOG_DIR/${RUN_NAME}_worker_${whost}.log 2>&1 &"
done

echo "All processes started. Collecting logs to logs/${RUN_NAME}_all.log"

# Stream consolidated log from root PS (it receives most messages)
ssh $SSH_FLAGS "$ROOT_ADDR" "tail -f $LOG_DIR/${RUN_NAME}_root.log" 2>/dev/null | \
    tee "logs/${RUN_NAME}_all.log" \
        >(grep --line-buffered "EVAL"    > "logs/${RUN_NAME}_eval.log") \
        >(grep --line-buffered "METRICS" > "logs/${RUN_NAME}_metrics.log") &

wait
