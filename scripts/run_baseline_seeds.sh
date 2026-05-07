#!/bin/bash
# -----------------------------------------------------------------------------
# Run multiple PFC simulations with different random seeds.
#
# Each seed produces an independent simulation in its own timestamped
# subdirectory under saved_simulations/. After each simulation, plot.py is
# called to generate all figures from the pickled output.
#
# Usage:
#   bash scripts/run_baseline_seeds.sh
#
# To generate a fresh set of seeds in Python:
#   python -c "import numpy as np; print([np.random.randint(10**7) for _ in range(20)])"
# -----------------------------------------------------------------------------

set -euo pipefail

# Add or remove seeds as needed. A single seed is provided here as a
# minimal working example.
SEEDS=(2835993)

counter=1
for seed in "${SEEDS[@]}"; do
    echo "----------------------------------------"
    echo "Trial ${counter} / ${#SEEDS[@]}  (seed=${seed})"
    echo "----------------------------------------"

    # Run the simulation. The seed argument overrides the value in
    # configuration_run_nest.yaml. Pickled output is written to
    # saved_simulations/<seed>_<timestamp>/sim_data.pkl.
    python3 -m model.run "${seed}"

    # Locate the most recent simulation directory and plot from it.
    latest_dir=$(ls -td saved_simulations/*/ | head -n 1)
    python3 -m plotting.plot "${latest_dir}sim_data.pkl"

    ((counter++))
done

echo "All ${#SEEDS[@]} trial(s) complete."
