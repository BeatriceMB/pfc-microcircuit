#!/usr/bin/env python3
"""
PV perturbation experiment analysis and figure generation.

Reads results/pv_results.csv produced by run_pv_experiment.py and generates
summary figures for the dissertation.

Usage:
    python3 -m experiments.plot_pv_results [path/to/pv_results.csv]

Defaults to results/pv_results.csv if no path is given.

Produces (in pv_figures/summary/):
  - exp1_peak_power_by_condition.png   — Experiment 1: pair-type degradation
  - exp1_peak_frequency.png
  - exp1_coherence.png
  - exp1_pn_participation.png
  - exp2_global_mechanisms.png         — Experiment 2: global degradation
  - exp3_neuron_vulnerability.png      — Experiment 3: peripheral-first vs random
  - coherence_spectra_overlay.png
  - coherence_spectra_by_experiment.png
  - overview_all_conditions.png
  - exp_pv_loss_hub_first_rate.png     — PN/PV ablation (hub-first)
  - exp_pv_loss_random_order_rate.png  — PN/PV ablation (random)
  - exp_motif_pv_freq.png              — motif PV peak frequency
  - exp_motif_pn_rate.png              — motif PN mean rate
"""

import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pathlib

# ── Match plotting style from plot_pfc_output ─────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.45,
    'xtick.major.width': 0.45,
    'ytick.major.width': 0.45,
    'xtick.minor.width': 0.35,
    'ytick.minor.width': 0.35,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 2.6,
    'ytick.major.size': 2.6,
    'xtick.minor.size': 1.4,
    'ytick.minor.size': 1.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'legend.handlelength': 0.8,
    'legend.handletextpad': 0.22,
    'legend.borderaxespad': 0.2,
    'lines.linewidth': 0.8,
    'font.size': 14.4,
    'axes.labelsize': 16.2,
    'axes.titlesize': 16.2,
    'legend.fontsize': 13.5,
    'xtick.labelsize': 13.5,
    'ytick.labelsize': 13.5,
    'figure.dpi': 150,
    'svg.fonttype': 'none',
})

# ── Load data ──────────────────────────────────────────────────────────────
csv_file = sys.argv[1] if len(sys.argv) > 1 else "results/pv_results.csv"
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} rows from {csv_file}")
print(f"Conditions: {df['condition'].unique()}")
print(f"Seeds: {df['seed'].unique()}")
print(f"Levels: {sorted(df['level'].unique())}")

out_dir = pathlib.Path("pv_figures/summary")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Normalise to baseline ──────────────────────────────────────────────────
# For each seed, normalise peak power and coherence to that seed's baseline
baseline = df[df['condition'] == 'baseline'].set_index('seed')

for metric in ['peak_power_pn', 'peak_power_pv', 'coherence_pn_pv']:
    df[f'{metric}_norm'] = df.apply(
        lambda row: row[metric] / baseline.loc[row['seed'], metric]
        if row['seed'] in baseline.index and baseline.loc[row['seed'], metric] > 0
        else np.nan,
        axis=1
    )

# ── Helper: group by level ────────────────────────────────────────────────
def agg(df_sub, metric):
    """Group by level, compute mean across seeds (no error bars for n=1)."""
    grouped = df_sub.groupby('level')[metric].mean().reset_index()
    return grouped['level'].values, grouped[metric].values


def plot_metric_by_condition(conditions, metric, ylabel, title, filename,
                             normalised=False, colors=None, xlim_start=-0.05,
                             integer_yticks=False, figsize=(10, 6), y_offsets=None,
                             baseline_point=None):
    """Plot one metric vs degradation level for multiple conditions."""
    fig, ax = plt.subplots(figsize=figsize)
    col = metric + '_norm' if normalised else metric

    if colors is None:
        cmap = cm.get_cmap('tab10')
        colors = {c: cmap(i) for i, c in enumerate(conditions)}

    if y_offsets is None:
        y_offsets = {}

    for cond in conditions:
        sub = df[df['condition'] == cond]
        if len(sub) == 0:
            continue
        levels, means = agg(sub, col)
        means = means + y_offsets.get(cond, 0)
        if baseline_point is not None:
            levels = np.concatenate([[0], levels])
            means = np.concatenate([[baseline_point], means])
        ax.plot(levels, means, marker='o', linewidth=2, markersize=7,
                label=cond.replace('_', ' '), color=colors.get(cond))

    if normalised:
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='baseline')

    ax.set_xlabel('Degradation level')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.set_xlim(xlim_start, 1.05)
    if integer_yticks:
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Which motif matters more? (gap+reciprocal vs chem-only)
# ═══════════════════════════════════════════════════════════════════════════
exp1_conditions = ['gap_recip_chem', 'uni_chem_only', 'global_gap']
exp1_colors = {
    'gap_recip_chem': '#7F77DD',   # purple — dominant motif
    'uni_chem_only':  '#1D9E75',   # teal — chemical only
    'global_gap':     '#e39d38',   # orange — global gap scaling
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

exp1_freq_offsets = {'uni_chem_only': 0.3}

_bl = df[df['condition'] == 'baseline'].iloc[0]
exp1_baseline = {
    'peak_freq_pv':         _bl['peak_freq_pv'],
    'mean_rate_pv':         _bl['mean_rate_pv'],
    'coherence_pn_pv_norm': 1.0,
    'mean_rate_pn':         _bl['mean_rate_pn'],
}

for ax, metric, ylabel in [
    (axes[0, 0], 'peak_freq_pv',          'Peak frequency (Hz)'),
    (axes[0, 1], 'mean_rate_pv',          'PV mean firing rate (Hz)'),
    (axes[1, 0], 'coherence_pn_pv_norm',  'PN–PV coherence (normalised)'),
    (axes[1, 1], 'mean_rate_pn',          'PN mean firing rate (Hz)'),
]:
    bl_val = exp1_baseline[metric]
    for cond in exp1_conditions:
        sub = df[df['condition'] == cond]
        if len(sub) == 0:
            continue
        levels, means = agg(sub, metric)
        if metric == 'peak_freq_pv':
            means = means + exp1_freq_offsets.get(cond, 0)
        levels = np.concatenate([[0], levels])
        means = np.concatenate([[bl_val], means])
        ax.plot(levels, means, marker='o', linewidth=2, markersize=7,
                label=cond.replace('_', ' '), color=exp1_colors.get(cond))
    if 'norm' in metric:
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    if metric == 'peak_freq_pv':
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_xlabel('Degradation level')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)

fig.suptitle('Experiment 1: Which motif matters more?', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(out_dir / 'exp1_motif_comparison.png', bbox_inches='tight')
plt.close(fig)
print("  Saved exp1_motif_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Is it gap junctions specifically? (global gap scaling)
# ═══════════════════════════════════════════════════════════════════════════
exp2_conditions = ['global_gap']
exp2_colors = {
    'global_gap': '#378ADD',     # blue — electrical
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, metric, ylabel in [
    (axes[0, 0], 'peak_power_pv_norm', 'PV peak power (normalised)'),
    (axes[0, 1], 'peak_freq_pv',       'Peak frequency (Hz)'),
    (axes[1, 0], 'coherence_pn_pv_norm', 'PN–PV coherence (normalised)'),
    (axes[1, 1], 'mean_rate_pv',        'PV mean firing rate (Hz)'),
]:
    for cond in exp2_conditions:
        sub = df[df['condition'] == cond]
        if len(sub) == 0:
            continue
        levels, means = agg(sub, metric)
        ax.plot(levels, means, marker='o', linewidth=2, markersize=7,
                label=cond.replace('_', ' '), color=exp2_colors.get(cond))
    if 'norm' in metric:
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Degradation level')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)

fig.suptitle('Experiment 2: Global gap junction degradation', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(out_dir / 'exp2_global_mechanisms.png', bbox_inches='tight')
plt.close(fig)
print("  Saved exp2_global_mechanisms.png")

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Hub-first vs random vs peripheral
# ═══════════════════════════════════════════════════════════════════════════
exp3_conditions = ['hub_first', 'random_order']
exp3_colors = {
    'hub_first':    '#E24B4A',   # red — expect steepest decline
    'random_order': '#888780',   # gray — control
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

exp3_baseline = {
    'peak_power_pv_norm':   1.0,
    'peak_freq_pv':         _bl['peak_freq_pv'],
    'coherence_pn_pv_norm': 1.0,
    'pn_participation':     _bl['pn_participation'],
}

for ax, metric, ylabel in [
    (axes[0, 0], 'peak_freq_pv',         'Peak frequency (Hz)'),
    (axes[0, 1], 'peak_power_pv_norm',   'PV peak power (normalised)'),
    (axes[1, 0], 'coherence_pn_pv_norm', 'PN–PV coherence (normalised)'),
]:
    bl_val = exp3_baseline[metric]
    for cond in exp3_conditions:
        sub = df[df['condition'] == cond]
        if len(sub) == 0:
            continue
        levels, means = agg(sub, metric)
        levels = np.concatenate([[0], levels])
        means  = np.concatenate([[bl_val], means])
        ax.plot(levels, means, marker='o', linewidth=2, markersize=7,
                label=cond.replace('_', ' '), color=exp3_colors.get(cond))
    if 'norm' in metric:
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    if metric == 'peak_freq_pv':
        ax.axhline(40, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(30, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(0.02, 30, 'gamma band limit', color='gray', fontsize=9,
                va='bottom', ha='left', transform=ax.get_yaxis_transform())
    ax.set_xlabel('Fraction of PV neurons ablated')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)

# Bottom-right: PV and PN mean firing rates for random_order only
ax_rate = axes[1, 1]
sub_rnd = df[df['condition'] == 'random_order']
if not sub_rnd.empty:
    for metric, color, label, bl_val in [
        ('mean_rate_pv', '#0d3b6e', 'PV', _bl['mean_rate_pv']),
        ('mean_rate_pn', '#BF5985', 'PN', _bl['mean_rate_pn']),
    ]:
        levels, means = agg(sub_rnd, metric)
        levels = np.concatenate([[0], levels])
        means  = np.concatenate([[bl_val], means])
        ax_rate.plot(levels, means, marker='o', linewidth=2, markersize=7,
                     color=color, label=label)
ax_rate.set_xlabel('Fraction of PV neurons ablated')
ax_rate.set_ylabel('Mean firing rate (Hz)')
ax_rate.set_title('Random order: PN & PV rates')
ax_rate.legend()
ax_rate.set_xlim(-0.05, 1.05)

fig.suptitle('Experiment 3: Neuron-level vulnerability ordering', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(out_dir / 'exp3_neuron_vulnerability.png', bbox_inches='tight')
plt.close(fig)
print("  Saved exp3_neuron_vulnerability.png")

# ═══════════════════════════════════════════════════════════════════════════
# COHERENCE SPECTRA — overlay all conditions at highest degradation level
# ═══════════════════════════════════════════════════════════════════════════
all_conds = exp1_conditions + exp2_conditions + exp3_conditions
all_colors = {**exp1_colors, **exp2_colors, **exp3_colors}

npz_files = sorted(glob.glob("pv_figures/**/coherence_spectrum.npz", recursive=True))
print(f"\n  Found {len(npz_files)} coherence spectrum files")

if npz_files:
    spectra = {}
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        key = (int(d['seed']), str(d['condition']), float(d['level']))
        spectra[key] = {'freqs': d['freqs'], 'coherence': d['coherence']}

    seeds = sorted(set(k[0] for k in spectra.keys()))

    # ── Figure 1: All conditions at highest degradation level ──
    fig, ax = plt.subplots(figsize=(12, 6))

    # Baseline
    for seed in seeds:
        bl_key = (seed, 'baseline', 0.0)
        if bl_key in spectra:
            ax.plot(spectra[bl_key]['freqs'], spectra[bl_key]['coherence'],
                    color='black', linewidth=2, alpha=0.8, label='baseline')
            break

    # Each condition at highest non-zero level
    for cond in all_conds:
        cond_levels = sorted([k[2] for k in spectra if k[1] == cond and k[2] > 0])
        if not cond_levels:
            continue
        highest = cond_levels[-1]
        for seed in seeds:
            key = (seed, cond, highest)
            if key in spectra:
                ax.plot(spectra[key]['freqs'], spectra[key]['coherence'],
                        color=all_colors.get(cond, 'gray'), linewidth=1.8, alpha=0.85,
                        label=f"{cond.replace('_', ' ')} ({highest:.0%})")
                break

    ax.axvspan(30, 80, alpha=0.08, color='green')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Coherence')
    ax.set_title('PN–PV coherence spectra: baseline vs degraded conditions')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / 'coherence_spectra_overlay.png', bbox_inches='tight')
    plt.close(fig)
    print("  Saved coherence_spectra_overlay.png")

    # ── Figure 2: per-experiment coherence ──
    exp_groups = [
        ('Motif comparison', exp1_conditions, exp1_colors),
        ('Global gap',       exp2_conditions, exp2_colors),
        ('Neuron vulnerability', exp3_conditions, exp3_colors),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (title, conds, colors) in zip(axes, exp_groups):
        # baseline
        for seed in seeds:
            bl_key = (seed, 'baseline', 0.0)
            if bl_key in spectra:
                ax.plot(spectra[bl_key]['freqs'], spectra[bl_key]['coherence'],
                        color='black', linewidth=2, alpha=0.7, label='baseline')
                break

        for cond in conds:
            cond_levels = sorted([k[2] for k in spectra if k[1] == cond and k[2] > 0])
            for lvl in cond_levels:
                for seed in seeds:
                    key = (seed, cond, lvl)
                    if key in spectra:
                        alpha = 0.5 + 0.5 * (lvl / max(cond_levels))
                        ax.plot(spectra[key]['freqs'], spectra[key]['coherence'],
                                color=colors.get(cond, 'gray'),
                                linewidth=1.5, alpha=alpha,
                                label=f"{cond.replace('_', ' ')} {lvl:.0%}")
                        break

        ax.axvspan(30, 80, alpha=0.08, color='green')
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title(title)
        ax.legend(fontsize=7, loc='upper right')

    axes[0].set_ylabel('Coherence')
    fig.suptitle('PN–PV coherence spectra by experiment', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'coherence_spectra_by_experiment.png', bbox_inches='tight')
    plt.close(fig)
    print("  Saved coherence_spectra_by_experiment.png")
else:
    print("  No coherence_spectrum.npz files found; skipping spectra figures.")

# ═══════════════════════════════════════════════════════════════════════════
# OVERVIEW: All conditions — gamma power comparison
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
for cond in all_conds:
    sub = df[df['condition'] == cond]
    if len(sub) == 0:
        continue
    levels, means = agg(sub, 'peak_power_pv_norm')
    ax.plot(levels, means, marker='o', linewidth=2, markersize=6,
            label=cond.replace('_', ' '), color=all_colors.get(cond))

ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='baseline')
ax.set_xlabel('Degradation level')
ax.set_ylabel('PV peak power (normalised to baseline)')
ax.set_title('PV perturbation: peak oscillatory power across all conditions')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(bottom=0)
plt.tight_layout()
fig.savefig(out_dir / 'overview_all_conditions.png', bbox_inches='tight')
plt.close(fig)
print("  Saved overview_all_conditions.png")

# ═══════════════════════════════════════════════════════════════════════════
# SINGLE-SEED ABLATION & MOTIF PLOTS (PN + PV with bands)
# ═══════════════════════════════════════════════════════════════════════════

# Single-seed qualitative results
target_seed = df['seed'].unique()[0]
sub_single = df[df['seed'] == target_seed]

def add_band_background(ax, levels, bands, ymin, ymax):
    """Shade background by PN band label at each ablation level."""
    for lvl, band in zip(levels, bands):
        if band == 'gamma':
            color = (0.85, 1.0, 0.85)  # light green
        elif band == 'beta':
            color = (1.0, 0.85, 0.95)  # light pink
        elif band == 'alpha':
            color = (0.9, 0.9, 1.0)    # light blue
        else:
            continue
        ax.axvspan(lvl - 0.025, lvl + 0.025, color=color, alpha=0.7, linewidth=0)

# 1) Ablation study: hub_first and random_order, PN & PV mean rate vs level
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

_bl_pv_rate = _bl['mean_rate_pv']
_bl_pn_rate = _bl['mean_rate_pn']

for ax, cond, title in zip(axes,
                           ['hub_first', 'random_order'],
                           ['Hub-first PV loss', 'Random PV loss']):
    df_cond = sub_single[sub_single['condition'] == cond].copy()
    df_cond = df_cond[df_cond['level'] > 0].sort_values('level')

    if df_cond.empty:
        continue

    levels = df_cond['level'].values
    pn_rate = df_cond['mean_rate_pn'].values
    pv_rate = df_cond['mean_rate_pv'].values

    levels_plot  = np.concatenate([[0], levels])
    pv_rate_plot = np.concatenate([[_bl_pv_rate], pv_rate])
    pn_rate_plot = np.concatenate([[_bl_pn_rate], pn_rate])

    ymin = 0
    ymax = max(np.max(pv_rate_plot), np.max(pn_rate_plot)) * 1.1

    ax.plot(levels_plot, pv_rate_plot, marker='o', linewidth=2, markersize=7,
            color='#0d3b6e', label='PV')
    ax.plot(levels_plot, pn_rate_plot, marker='o', linewidth=2, markersize=7,
            color='#BF5985', label='PN')

    ax.set_xlabel('PV ablation level')
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(levels_plot)
    ax.legend(loc='best')

axes[0].set_ylabel('Mean firing rate (Hz)')

fig.tight_layout()
fig.savefig(out_dir / 'exp_pv_loss_rate.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved exp_pv_loss_rate.png")

# 2) Connectivity motif: bar plots (PV peak freq, PN mean rate)
motif_conds = ['baseline', 'gap_recip_chem', 'uni_chem_only', 'global_gap']
df_motif = sub_single[sub_single['condition'].isin(motif_conds)].copy()

order = ['baseline', 'gap_recip_chem', 'uni_chem_only', 'global_gap']
labels = ['Baseline', 'Gap+recip', 'Uni chem-only', 'Global gap']
colors = ['#4c4c4c', '#7F77DD', '#1D9E75', '#e39d38']

# PV peak frequency
fig, ax = plt.subplots(figsize=(4.0, 3.2))
frequencies = []
for cond in order:
    rows = df_motif[df_motif['condition'] == cond].sort_values('level')
    if rows.empty:
        frequencies.append(np.nan)
    else:
        row = rows.iloc[-1]
        frequencies.append(row['peak_freq_pv'])

x = np.arange(len(order))
ax.bar(x, frequencies, color=colors, width=0.62)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel('PV peak frequency (Hz)')
ax.set_title('PV motif degradation')
fig.tight_layout()
fig.savefig(out_dir / 'exp_motif_pv_freq.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved exp_motif_pv_freq.png")

# PN mean rate
fig, ax = plt.subplots(figsize=(4.0, 3.2))
rates = []
for cond in order:
    rows = df_motif[df_motif['condition'] == cond].sort_values('level')
    if rows.empty:
        rates.append(np.nan)
    else:
        row = rows.iloc[-1]
        rates.append(row['mean_rate_pn'])

ax.bar(x, rates, color=colors, width=0.62)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel('PN mean rate (Hz)')
ax.set_title('PN firing under motif degradation')
fig.tight_layout()
fig.savefig(out_dir / 'exp_motif_pn_rate.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved exp_motif_pn_rate.png")

print(f"\nAll figures saved to {out_dir}/")