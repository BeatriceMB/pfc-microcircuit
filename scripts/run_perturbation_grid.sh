#!/bin/bash
#
# Perturbation Experiment Grid Runner
#
# Runs baseline once per seed, then each motif degradation condition at
# two degradation levels and each ablation condition at seven levels.
# Results accumulate in results/pv_results.csv (one row per run).
#
# Usage:
#   bash scripts/run_perturbation_grid.sh
#

set -euo pipefail

seeds=(9141774)


# Synaptic degradation ----------------------------------------------------------------
    # Experiments 1–2
main_levels=(0.5 1.0)
main_conditions=(gap_recip_chem uni_chem_only global_gap)


# Neuron loss ---------------------------------------------------------------------------
    # Experiment 3
exp3_levels=(0.1 0.2 0.4 0.5 0.6 0.8 1.0)
exp3_conditions=(hub_first random_order)

actual_runs=$(( ${#seeds[@]} * (
    1
    + ${#main_conditions[@]} * ${#main_levels[@]}
    + ${#exp3_conditions[@]} * ${#exp3_levels[@]}
) ))

echo "═══════════════════════════════════════════════════════════════"
echo " Perturbation Experiment Grid"
echo " Seeds: ${#seeds[@]} (${seeds[*]})"
echo " Main conditions: ${#main_conditions[@]} (${main_conditions[*]})"
echo " Main levels: ${#main_levels[@]} (${main_levels[*]})"
echo " Experiment 3 conditions: ${#exp3_conditions[@]} (${exp3_conditions[*]})"
echo " Experiment 3 levels: ${#exp3_levels[@]} (${exp3_levels[*]})"
echo " Total runs: ${actual_runs}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

counter=0
failed=0

for seed in "${seeds[@]}"; do
    ((counter++))
    echo "[$counter/$actual_runs] seed=$seed BASELINE"
    python3 -m experiments.run_pv_experiment "$seed" baseline 0.0
    if [ $? -ne 0 ]; then
        echo " *** FAILED ***"
        ((failed++))
    fi
    echo ""

    for condition in "${main_conditions[@]}"; do
        for level in "${main_levels[@]}"; do
            ((counter++))
            echo "[$counter/$actual_runs] seed=$seed $condition level=$level"
            python3 -m experiments.run_pv_experiment "$seed" "$condition" "$level"
            if [ $? -ne 0 ]; then
                echo " *** FAILED ***"
                ((failed++))
            fi
            echo ""
        done
    done

    for condition in "${exp3_conditions[@]}"; do
        for level in "${exp3_levels[@]}"; do
            ((counter++))
            echo "[$counter/$actual_runs] seed=$seed $condition level=$level"
            python3 -m experiments.run_pv_experiment "$seed" "$condition" "$level"
            if [ $? -ne 0 ]; then
                echo " *** FAILED ***"
                ((failed++))
            fi
            echo ""
        done
    done
done

echo "═══════════════════════════════════════════════════════════════"
echo " Complete: $counter runs, $failed failed"
echo " Results in: results/pv_results.csv"
echo "═══════════════════════════════════════════════════════════════"