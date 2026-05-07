# Multilayer PFC Microcircuit Model

A biologically constrained spiking neural network model of prefrontal
cortex (PFC) layers 4 and 5, built in NEST. The model expresses a
PING-regime gamma oscillation in L5 and beta-range activity in L4,
and provides a framework for systematic perturbation of PV interneuron
connectivity motifs and cell counts in the context of neurodegenerative
pathology.

This repository contains the full pipeline used in:

> Baldi, B. M. (2026). *A biologically constrained spiking model of
> prefrontal layers 4 and 5 microcircuitry: characterising
> connectivity-motif specific perturbations of PV interneuron networks
> in neurodegeneration.* BSc Dissertation, University of St Andrews.

---

## Quick start

After installation (see below), the entire pipeline is driven by two
shell scripts run from the repository root:

```bash
# Reproduce the baseline characterisation across multiple seeds
bash scripts/run_baseline_seeds.sh

# Reproduce the full motif-degradation and PV-ablation experiments
bash scripts/run_perturbation_grid.sh
```

Each script handles simulation, analysis, and figure generation
end-to-end. Results are written to timestamped output folders
described below.

---

## Repository structure

```
pfc-microcircuit/
├── README.md
├── environment.yml                Conda environment specification
├── requirements.txt               Pip-only dependency list
├── configuration_run_nest.yaml    Simulation parameters (seed, duration, plots)
│
├── model/                         Network model
│   ├── parameters.py              Network parameters and derived quantities
│   ├── network.py                 Network construction (populations, synapses)
│   ├── simulation.py              NEST kernel initialisation
│   └── run.py                     Single-simulation entry point
│
├── analysis/                      Post-simulation analysis
│   ├── metrics.py                 Spectral metrics (PSD, peak frequency, coherence)
│   └── population.py              Population-rate construction, spike processing
│
├── plotting/                      Baseline figure generation
│   └── plot.py                    Reads pickled simulation, produces figures
│
├── experiments/                   Perturbation experiments
│   ├── run_pv_experiment.py      Single perturbation run (one condition × level)
│   ├── plot_pv_experiment.py     Per-condition figures
│   └── plot_pv_results.py        Cross-condition summary figures
│
├── scripts/                       Batch runners
│   ├── run_baseline_seeds.sh      Baseline across multiple seeds
│   └── run_perturbation_grid.sh   Full perturbation grid
│
└── installer_files/               NESTML neuron specification
    ├── aeif_cond_beta.nestml
    └── create_gap_enabled_aeif.py
```

---

## Installation

The model depends on NEST (≥ 3.7) and NESTML (≥ 7.0), both installed
via conda. The recommended path uses the supplied `environment.yml`:

```bash
git clone https://github.com/BeatriceMB/pfc-microcircuit.git
cd pfc-microcircuit

conda env create -f environment.yml
conda activate nest-pfc
```

If you prefer a manual installation:

- NEST: https://nest-simulator.readthedocs.io/en/stable/installation/
- NESTML: https://nestml.readthedocs.io/en/latest/installation.html

After activating the environment, compile the custom AdEx neuron with
gap junction support:

```bash
cd installer_files
python create_gap_enabled_aeif.py
cd ..
```

This builds the `nestml_gap_aeif_beta_module` and registers
`aeif_cond_beta_nestml` as an available neuron model. The build step is
required only once per environment.

---

## Running the model

All commands below are executed from the repository root.

### Single baseline simulation

```bash
python -m model.run
```

The simulation reads parameters from `configuration_run_nest.yaml` and
writes a pickle of the spike data and membrane potential traces to
`saved_simulations/<seed>_<timestamp>/sim_data.pkl`. To override the
seed defined in the YAML:

```bash
python -m model.run 2835993
```

### Plotting baseline figures

To generate every baseline figure for the most recent simulation:

```bash
python -m plotting.plot
```

To plot a specific simulation:

```bash
python -m plotting.plot saved_simulations/<seed>_<timestamp>/sim_data.pkl
```

Figures are written to a `Figures/` subdirectory next to the pickle file.

### Multi-seed baseline characterisation

```bash
bash scripts/run_baseline_seeds.sh
```

This runs the baseline simulation across every seed listed in the
script's `SEEDS` array, generating the full set of baseline figures for
each. Edit the `SEEDS` array at the top of the script to control which
seeds are run.

### Single perturbation experiment

```bash
python -m experiments.run_pv_experiment <seed> <condition> <level>
```

Where:

- `seed` is an integer (controls network topology)
- `condition` is one of:
  - `baseline` — no perturbation
  - `gap_recip_chem` — degrade gap + reciprocal chemical PV-PV pairs
  - `gap_uni_chem` — degrade gap + unidirectional chemical pairs
  - `uni_chem_only` — degrade unidirectional chemical-only pairs
  - `gap_only` — degrade gap-only pairs
  - `global_gap` — scale down all PV-PV gap junction weights
  - `global_chem` — scale down all PV-PV chemical weights
  - `hub_first` — ablate PV neurons from most- to least-connected
  - `random_order` — ablate PV neurons in random order
  - `peripheral_first` — ablate PV neurons from least- to most-connected
- `level` is a fraction in [0.0, 1.0] (0 = intact, 1 = fully degraded)

Example:

```bash
python -m experiments.run_pv_experiment 9141774 gap_recip_chem 0.5
```

Each run appends a row to `results/pv_results.csv` and produces a set
of per-condition figures in `pv_figures/<seed>/<condition>_<level>/`.

### Full perturbation grid

```bash
bash scripts/run_perturbation_grid.sh
```

This runs the full experimental matrix used in the dissertation:
baseline plus three motif degradation conditions at two levels and two
ablation conditions at seven levels — 21 simulations per seed. Edit the
`seeds` array at the top of the script to control which seeds are run.

### Cross-condition summary figures

After the perturbation grid has finished:

```bash
python -m experiments.plot_pv_results
```

Reads `results/pv_results.csv` and produces summary figures comparing
conditions across the perturbation grid (peak power, peak frequency,
coherence, firing rates). Output figures are written to
`pv_figures/summary/`.

---

## Configuration

All simulation parameters are controlled through
`configuration_run_nest.yaml`. The most commonly adjusted are:

```yaml
seed: 2835993             # Random seed (override on CLI for python -m model.run)
t_steps: 5000.0           # Simulation duration in ms
delta_clock: 0.1          # Time resolution in ms
save_results: True        # Write outputs to saved_simulations/
spatial_sparsity: 1       # 0 = flat connection probability, 1 = distance-dependent
spatial_weight: 1         # 0 = uniform synaptic weights, 1 = distance-dependent
gap_junctions_enabled: 1  # 0 = disable gap junctions, 1 = enable
```

Plot toggles in the same file control which baseline figures are generated.

---

## Output directory structure

After running the full pipeline, the repository will contain four
generated output folders:

```
pfc-microcircuit/
├── saved_simulations/           Per-seed baseline simulation outputs
│   └── <seed>_<timestamp>/
│       ├── args_<id>.yaml       Configuration used for this run
│       ├── sim_data.pkl         Pickled simulation data
│       └── Figures/             All baseline figures
│
├── results/                     Aggregated perturbation results
│   └── pv_results.csv          One row per (seed, condition, level)
│
├── pv_plot_data/               Per-condition npz files
│   └── pv_plot_data_<seed>_<condition>_<level>.npz
│
└── pv_figures/                 Perturbation figures
    ├── <seed>/
    │   ├── baseline_0.0/
    │   ├── gap_recip_chem_0.5/
    │   └── ...
    └── summary/                 Cross-condition summary figures
```

---

## Model overview

**Populations modelled**

| Population | Layer | Type            | Role                              |
|------------|-------|-----------------|-----------------------------------|
| PN L4      | L4    | Pyramidal       | Thalamocortical input integration |
| SST L4     | L4    | Inhibitory      | Distal-dendritic inhibition       |
| PN L5      | L5    | Pyramidal       | Cortical output                   |
| PV L5      | L5    | Fast-spiking    | Perisomatic inhibition, gamma     |

All neurons use the AdEx formalism with beta-function synapses
implemented via NESTML.

**PV-PV connectivity**

PV-PV pairs are wired using a two-stage probabilistic model fitted to
Galarreta and Hestrin (2002, *PNAS*) Table 1: a distance-dependent
connectivity gate followed by data-driven categorical assignment of
pair type. Six joint categories are preserved (gap + reciprocal
chemical, gap + unidirectional chemical, gap only, reciprocal chemical
only, unidirectional chemical only, neither), reproducing the empirical
co-regulation of gap junctions and chemical synapses.

**PV-PN connectivity**

PV ↔ PN reciprocal-overrepresentation is implemented per-pair using
empirical probabilities from the paired-recording literature.

**Other pathways**

All remaining pathways use distance-dependent pairwise-Bernoulli
connectivity with biologically motivated decay constants and peak
probabilities. Full parameterisation and references are given in
`model/parameters.py`.

---

## Citation

If you use this model in published work, please cite:

```bibtex
@thesis{baldi2026pfc,
  author = {Baldi, Beatrice Maria},
  title  = {A biologically constrained spiking model of prefrontal
            layers 4 and 5 microcircuitry: characterising
            connectivity-motif specific perturbations of PV interneuron
            networks in neurodegeneration},
  school = {University of St Andrews},
  year   = {2026}
}
```

---

## Author

**Beatrice Maria Baldi**

For questions, bug reports, or contributions, please open an issue on
the GitHub repository.

---

## License
MIT 
