#!/usr/bin/env bash
# Launch all three baselines × N seeds, sequentially (so the M4 stays
# under its thermal envelope). Run from inside `code/` after activating
# the venv created by setup.sh.
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1 2 3 4"}
CONFIGS=(
    "configs/ppo_lunarlander.yaml"
    "configs/sac_lunarlander.yaml"
    "configs/ppo_dr_lunarlander.yaml"
)

mkdir -p logs

for cfg in "${CONFIGS[@]}"; do
    for seed in ${SEEDS}; do
        tag="$(basename "${cfg}" .yaml)__seed${seed}"
        log="logs/${tag}.log"
        echo "[run] ${tag} → ${log}"
        python -m src.train --config "${cfg}" --seed "${seed}" 2>&1 | tee "${log}"
        # cool-down between heavy runs (fanless M4)
        sleep 10
    done
done

echo "[run] all baselines complete. Plot via:"
echo "    python scripts/plot_results.py --runs-root runs/"
