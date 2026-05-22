#!/usr/bin/env bash

CFGDIR="configs"
LOGDIR="logs"
TRIALS=1

mkdir -p "$LOGDIR"

configs=(
    "6-6-6_ss_control"
    "6-6-6_ss_homog"
    "6-6-6_ss_heter"
    "6-6-6_as_control"
    "6-6-6_as_homog"
    "6-6-6_as_heter"
    "6-6-6_aa_control"
    "6-6-6_aa_homog"
    "6-6-6_aa_heter"
    "6-6-6_sa_control"
    "6-6-6_sa_homog"
    "6-6-6_sa_heter"
)

for ((trial=1; trial<=TRIALS; trial++)); do
    echo "=== Trial $trial ==="

    for cfg in "${configs[@]}"; do
        name="${cfg}"

        echo "Running $cfg (trial $trial)..."
        ./run.sh "${CFGDIR}/${cfg}.yaml" "${name}_test" > output.log
    done
done

echo "All runs complete."
