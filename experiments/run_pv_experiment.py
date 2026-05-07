#!/usr/bin/env python3
"""
PV interneuron perturbation experiment runner.

Runs a single perturbation experiment for the multilayer PFC model:
either motif-selective synaptic degradation or progressive PV cell
ablation, controlled by the (condition, level) command-line arguments.

Usage:
    python3 -m experiments.run_pv_experiment <seed> <condition> <level>

Arguments:
    seed       : integer RNG seed (controls network topology)
    condition  : degradation condition (see CONDITIONS below)
    level      : degradation fraction 0.0–1.0 (0 = intact, 1 = fully degraded)

Conditions:
    baseline           — no degradation (control)
    gap_recip_chem     — degrade gap + reciprocal chemical pairs
    gap_uni_chem       — degrade gap + unidirectional chemical pairs
    uni_chem_only      — degrade unidirectional chemical-only pairs
    gap_only           — degrade gap-only pairs
    global_gap         — scale down ALL gap junction weights
    global_chem        — scale down ALL chemical PV–PV weights
    hub_first          — ablate PV neurons from most- to least-connected
    random_order       — ablate PV neurons in random order
    peripheral_first   — ablate PV neurons from least- to most-connected

Outputs:
    Appends one row to results/pv_results.csv with all outcome measures.
    Saves per-condition plot data to pv_plot_data/.
    Saves figures to pv_figures/<seed>/<condition>_<level>/.

Example:
    python3 -m experiments.run_pv_experiment 2835993 gap_recip_chem 0.4
"""

import nest
import numpy as np
import sys
import time
import csv
import pathlib
from scipy.signal import welch, coherence
from scipy.ndimage import gaussian_filter

# ── Parse arguments ──────────────────────────────────────────────────────
if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)

SEED      = int(sys.argv[1])
CONDITION = sys.argv[2]
LEVEL     = float(sys.argv[3])

CONDITIONS = [
    'baseline', 'gap_recip_chem', 'gap_uni_chem', 'uni_chem_only',
    'gap_only', 'global_gap', 'global_chem',
    'hub_first', 'random_order', 'peripheral_first',
]
assert CONDITION in CONDITIONS, f"Unknown condition '{CONDITION}'. Must be one of {CONDITIONS}"
assert 0.0 <= LEVEL <= 1.0, f"Level must be 0.0–1.0, got {LEVEL}"

print(f"\n{'='*60}")
print(f"  PV Experiment: seed={SEED}  condition={CONDITION}  level={LEVEL}")
print(f"{'='*60}\n")

# ── Patch sys.argv so model.parameters reads the seed ───────────────────
sys.argv = [sys.argv[0], str(SEED)]

# ── Initialise NEST ──────────────────────────────────────────────────────
from model import simulation as ss
from model import parameters as netparams
from analysis import population as popfunc

ss.nest_start()
nn = netparams.neural_network()

# Set numpy seed to match NEST seed → identical topology across conditions
np.random.seed(SEED)

# ── Build network ────────────────────────────────────────────────────────
from model import network as pfc
pfc1 = pfc.create_pfc_network()

pair_types = pfc1.pv_pv_pair_types
profiles   = pfc1.pv_profiles
pv_ids     = pair_types["pv_ids"]


# ═══════════════════════════════════════════════════════════════════════════
# DEGRADATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _zero_pair_connections(mask_key, level, include_gap=True, include_chem=True):
    """
    Zero out connections for a fraction of pairs in a given category.

    Parameters
    ----------
    mask_key     : str — key into pair_types dict (e.g. 'gap_recip_chem_mask')
    level        : float — fraction of pairs to degrade (0–1)
    include_gap  : bool — whether to zero gap junctions
    include_chem : bool — whether to zero chemical synapses
    """
    mask = pair_types[mask_key]
    ii, jj = np.where(mask)
    n_pairs = len(ii)
    n_remove = int(n_pairs * level)

    if n_remove == 0:
        return 0

    remove_idx = np.random.choice(n_pairs, n_remove, replace=False)

    n_zeroed = 0
    for idx in remove_idx:
        src_id = int(pv_ids[ii[idx]])
        tgt_id = int(pv_ids[jj[idx]])

        if include_chem:
            # Chemical: check both directions (reciprocal pairs have i→j and j→i)
            for s, t in [(src_id, tgt_id), (tgt_id, src_id)]:
                conns = nest.GetConnections(source=nest.NodeCollection([s]),
                                            target=nest.NodeCollection([t]),
                                            synapse_model="static_synapse")
                if conns:
                    conns.weight = [0.0] * len(conns)
                    n_zeroed += len(conns)

        if include_gap:
            for s, t in [(src_id, tgt_id), (tgt_id, src_id)]:
                conns = nest.GetConnections(source=nest.NodeCollection([s]),
                                            target=nest.NodeCollection([t]),
                                            synapse_model="gap_junction")
                if conns:
                    conns.weight = [0.0] * len(conns)
                    n_zeroed += len(conns)

    return n_zeroed


def _global_scale_weights(synapse_model, scale_factor):
    """Scale all PV–PV connections of a given synapse model by a factor."""
    pv_node_collection = pfc1.pv_inh_l5
    conns = nest.GetConnections(source=pv_node_collection,
                                target=pv_node_collection,
                                synapse_model=synapse_model)
    if conns:
        old_weights = np.array(conns.weight)
        conns.weight = (old_weights * scale_factor).tolist()
    return len(conns) if conns else 0


def _ablate_neurons(neuron_ids):
    """
    Fully ablate PV neurons: cells remain as NEST objects but are completely
    electrically isolated from the network.

    Unlike functional silencing (which only removes the cell's output),
    ablation removes all network influence on AND from the cell:
      - Tonic external drive (I_e) set to 0
      - All outgoing connections zeroed (chemical synapses, gap junctions)
      - All incoming connections zeroed (chemical synapses from PN/SST/other PV,
        Poisson background input, incoming half of gap junctions)

    The cell continues to exist in NEST (preserving neuron indices and allowing
    spike detectors to remain attached) but contributes no synaptic input to any
    partner and receives none itself, modelling complete loss of function as
    occurs with PV cell death in neurodegeneration.
    """
    for nid in neuron_ids:
        node = nest.NodeCollection([nid])

        # 1. Remove tonic external drive
        nest.SetStatus(node, {"I_e": 0.0})

        # 2. Zero all OUTGOING connections (any synapse type)
        #    Covers: PV→PN chemical, PV→PV chemical, outgoing gap junctions
        out_conns = nest.GetConnections(source=node)
        if out_conns:
            out_conns.weight = [0.0] * len(out_conns)

        # 3. Zero all INCOMING connections (any synapse type, any source)
        #    Covers: PN→PV, SST→PV, other-PV→PV, Poisson→PV, incoming gap junctions
        in_conns = nest.GetConnections(target=node)
        if in_conns:
            in_conns.weight = [0.0] * len(in_conns)


def _neuron_order_by_connectivity(order='hub_first'):
    """Return PV neuron IDs sorted by total connectivity."""
    sorted_items = sorted(profiles.items(),
                          key=lambda x: x[1]["total_connected"],
                          reverse=(order == 'hub_first'))
    return [int(k) for k, v in sorted_items]


# ── Apply degradation ────────────────────────────────────────────────────
t_degrade = time.perf_counter()
n_affected = 0

if CONDITION == 'baseline':
    print("  No degradation (baseline control)")

elif CONDITION == 'gap_recip_chem':
    n_affected = _zero_pair_connections('gap_recip_chem_mask', LEVEL,
                                        include_gap=True, include_chem=True)
    print(f"  Degraded gap+reciprocal chem pairs: {n_affected} connections zeroed")

elif CONDITION == 'gap_uni_chem':
    n_affected = _zero_pair_connections('gap_uni_chem_mask', LEVEL,
                                        include_gap=True, include_chem=True)
    print(f"  Degraded gap+unidirectional chem pairs: {n_affected} connections zeroed")

elif CONDITION == 'uni_chem_only':
    n_affected = _zero_pair_connections('uni_chem_only_mask', LEVEL,
                                        include_gap=False, include_chem=True)
    print(f"  Degraded unidirectional chem-only pairs: {n_affected} connections zeroed")

elif CONDITION == 'gap_only':
    n_affected = _zero_pair_connections('gap_only_mask', LEVEL,
                                        include_gap=True, include_chem=False)
    print(f"  Degraded gap-only pairs: {n_affected} connections zeroed")

elif CONDITION == 'global_gap':
    scale = 1.0 - LEVEL
    n_affected = _global_scale_weights("gap_junction", scale)
    print(f"  Global gap junction scaling: ×{scale:.2f} on {n_affected} connections")

elif CONDITION == 'global_chem':
    scale = 1.0 - LEVEL
    n_affected = _global_scale_weights("static_synapse", scale)
    print(f"  Global chemical PV–PV scaling: ×{scale:.2f} on {n_affected} connections")

elif CONDITION in ('hub_first', 'random_order', 'peripheral_first'):
    if CONDITION == 'random_order':
        order = list(pv_ids.astype(int))
        np.random.shuffle(order)
    elif CONDITION == 'hub_first':
        order = _neuron_order_by_connectivity('hub_first')
    else:
        order = _neuron_order_by_connectivity('peripheral_first')

    n_ablate = int(len(order) * LEVEL)
    to_ablate = order[:n_ablate]
    _ablate_neurons(to_ablate)
    n_affected = n_ablate
    print(f"  Ablated {n_ablate} PV neurons ({CONDITION})")

print(f"  Degradation applied in {time.perf_counter() - t_degrade:.1f}s\n")


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATE
# ═══════════════════════════════════════════════════════════════════════════
init_time = 50
nest.Simulate(init_time)
num_steps = int(nn.sim_time / nn.time_resolution)
t_start = time.perf_counter()
for i in range(int(num_steps / 10) - init_time):
    nest.Simulate(nn.time_resolution * 10)
    print(f"  t = {nest.biological_time:.0f} ms", end="\r")
t_sim = time.perf_counter() - t_start
print(f"\n  Simulation complete in {t_sim:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
# READ SPIKE DATA
# ═══════════════════════════════════════════════════════════════════════════
_, spiketimes_pn_l5 = popfunc.read_spike_data(pfc1.spike_detector_pn_exc_l5)
_, spiketimes_pv_l5 = popfunc.read_spike_data(pfc1.spike_detector_pv_inh_l5)


# ═══════════════════════════════════════════════════════════════════════════
# OUTCOME MEASURES
# ═══════════════════════════════════════════════════════════════════════════

def pop_rate(spiketimes, n_neurons, bin_ms=1.0, smooth_ms=5.0):
    """Population mean firing rate (Hz), smoothed."""
    bins = np.arange(0, nn.sim_time + bin_ms, bin_ms)
    counts = np.zeros(len(bins) - 1)
    for i in range(n_neurons):
        c, _ = np.histogram(spiketimes[0][i], bins)
        counts += c
    rate = counts / (n_neurons * (bin_ms / 1000.0))
    if smooth_ms > 0:
        rate = gaussian_filter(rate, smooth_ms / bin_ms)
    t = 0.5 * (bins[:-1] + bins[1:])
    return rate, t


def spectral_peak(rate_signal, fs=1000.0, f_low=2.0, f_high=150.0, half_width=5.0):
    """Peak frequency and integrated power ±5 Hz around peak from Welch PSD."""
    freqs, psd = welch(rate_signal, fs=fs, nperseg=min(2048, len(rate_signal)))
    mask = (freqs >= f_low) & (freqs <= f_high)
    peak_idx = np.argmax(psd[mask])
    peak_freq = freqs[mask][peak_idx]
    # Integrate in a window around the peak
    band_mask = (freqs >= peak_freq - half_width) & (freqs <= peak_freq + half_width)
    peak_power = np.trapz(psd[band_mask], freqs[band_mask])
    return peak_freq, peak_power


def classify_band(freq):
    """Classify a frequency into its oscillation band."""
    if freq < 4:
        return 'delta'
    elif freq < 8:
        return 'theta'
    elif freq < 13:
        return 'alpha'
    elif freq < 30:
        return 'beta'
    elif freq < 80:
        return 'gamma'
    else:
        return 'high-gamma'


def gamma_coherence(sig1, sig2, fs=1000.0, f_low=30.0, f_high=80.0):
    """Mean-squared coherence between two signals in gamma band."""
    freqs, coh = coherence(sig1, sig2, fs=fs, nperseg=min(1024, len(sig1)))
    mask = (freqs >= f_low) & (freqs <= f_high)
    return np.mean(coh[mask]) if mask.sum() > 0 else 0.0


def pn_participation(spiketimes_pn, n_pn, osc_freq, sim_time, min_cycle_frac=0.1):
    """Fraction of PN neurons that fire in ≥min_cycle_frac of gamma cycles."""
    if osc_freq <= 0:
        return 0.0
    cycle_ms = 1000.0 / osc_freq
    cycle_edges = np.arange(0, sim_time + cycle_ms, cycle_ms)
    n_cycles = len(cycle_edges) - 1
    if n_cycles == 0:
        return 0.0

    participating = np.zeros(n_pn, dtype=bool)
    for i in range(n_pn):
        spikes = spiketimes_pn[0][i]
        if len(spikes) == 0:
            continue
        cycles_with_spike = np.unique(np.searchsorted(cycle_edges[1:], spikes))
        frac = len(cycles_with_spike) / n_cycles
        participating[i] = frac >= min_cycle_frac
    return participating.mean()


def mean_firing_rate(spiketimes, n_neurons, sim_time):
    """Mean individual neuron firing rate (Hz)."""
    total = sum(len(spiketimes[0][i]) for i in range(n_neurons))
    return total / (n_neurons * sim_time / 1000.0)


# ── Compute metrics ──────────────────────────────────────────────────────
rate_pn, t_pn = pop_rate(spiketimes_pn_l5, nn.pn_exc_l5_count)
rate_pv, t_pv = pop_rate(spiketimes_pv_l5, nn.total_pv)

# Ensure same length for coherence
min_len = min(len(rate_pn), len(rate_pv))
rate_pn_c = rate_pn[:min_len]
rate_pv_c = rate_pv[:min_len]

peak_freq_pn, peak_pow_pn = spectral_peak(rate_pn)
peak_freq_pv, peak_pow_pv = spectral_peak(rate_pv)
band_pn = classify_band(peak_freq_pn)
band_pv = classify_band(peak_freq_pv)
coh_pn_pv     = gamma_coherence(rate_pn_c, rate_pv_c)
mfr_pn        = mean_firing_rate(spiketimes_pn_l5, nn.pn_exc_l5_count, nn.sim_time)
mfr_pv        = mean_firing_rate(spiketimes_pv_l5, nn.total_pv, nn.sim_time)
pn_part       = pn_participation(spiketimes_pn_l5, nn.pn_exc_l5_count,
                                  peak_freq_pv, nn.sim_time)

# Full coherence spectrum (saved for overlay plotting)
coh_freqs, coh_spectrum = coherence(rate_pn_c, rate_pv_c, fs=1000.0,
                                     nperseg=min(1024, len(rate_pn_c)))

print(f"\n── Outcome measures ──────────────────────────────────────────")
print(f"  Peak freq    (PN): {peak_freq_pn:.1f} Hz  [{band_pn}]")
print(f"  Peak freq    (PV): {peak_freq_pv:.1f} Hz  [{band_pv}]")
print(f"  Peak power   (PN): {peak_pow_pn:.4f}")
print(f"  Peak power   (PV): {peak_pow_pv:.4f}")
print(f"  Coherence PN–PV:   {coh_pn_pv:.4f}")
print(f"  Mean rate    (PN): {mfr_pn:.2f} Hz")
print(f"  Mean rate    (PV): {mfr_pv:.2f} Hz")
print(f"  PN participation:  {pn_part:.3f}")
print(f"{'─'*60}")


# ═══════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════
results_dir = pathlib.Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
results_file = results_dir / "pv_results.csv"
header = [
    "seed", "condition", "level", "n_affected",
    "peak_freq_pn", "peak_freq_pv",
    "peak_power_pn", "peak_power_pv",
    "band_pn", "band_pv",
    "coherence_pn_pv",
    "mean_rate_pn", "mean_rate_pv",
    "pn_participation",
    "sim_time_s",
]
row = [
    SEED, CONDITION, LEVEL, n_affected,
    round(peak_freq_pn, 2), round(peak_freq_pv, 2),
    round(peak_pow_pn, 6), round(peak_pow_pv, 6),
    band_pn, band_pv,
    round(coh_pn_pv, 6),
    round(mfr_pn, 4), round(mfr_pv, 4),
    round(pn_part, 4),
    round(t_sim, 1),
]

write_header = not pathlib.Path(results_file).exists()
with open(results_file, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    writer.writerow(row)

print(f"\n  Results appended to {results_file}")



# Save data for condition-specific plotting
plot_data_dir = pathlib.Path('pv_plot_data')
plot_data_dir.mkdir(parents=True, exist_ok=True)
plot_data_file = plot_data_dir / f"pv_plot_data_{SEED}_{CONDITION}_{LEVEL:.1f}.npz"
np.savez(
    plot_data_file,
    seed=SEED,
    condition=CONDITION,
    level=LEVEL,
    nn_sim_time=nn.sim_time,
    nn_total_pv=nn.total_pv,
    nn_pn_exc_l5_count=nn.pn_exc_l5_count,
    rate_pn=rate_pn,
    t_pn=t_pn,
    rate_pv=rate_pv,
    t_pv=t_pv,
    coh_freqs=coh_freqs,
    coh_spectrum=coh_spectrum,
    mfr_pn=mfr_pn,
    mfr_pv=mfr_pv,
    peak_freq_pn=peak_freq_pn,
    peak_freq_pv=peak_freq_pv,
    spiketimes_pn_l5=np.array(spiketimes_pn_l5[0], dtype=object),
    spiketimes_pv_l5=np.array(spiketimes_pv_l5[0], dtype=object),
)

# Trigger per-condition plotting on the file we just wrote.
from experiments.plot_pv_experiment import plot_one
plot_one(str(plot_data_file))


