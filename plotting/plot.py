#!/usr/bin/env python
"""
Plotting script for the multilayer PFC model.

Reads a pickled `sim_data` file produced by `model/run.py` and
generates every figure used in the dissertation. No simulation is
run here, so replotting from cached data is fast.

Usage
-----
    # Plot the most recent simulation (default behaviour):
    python -m plotting.plot

    # Plot a specific simulation by path:
    python -m plotting.plot saved_simulations/<seed>_<timestamp>/sim_data.pkl

Figures are written to a `Figures/` subdirectory next to the pickle.

Figure groups
-------------
1. Per-population rate + heatmap
2. Stacked population oscillations
3. Power spectra (PV L5 and PN L5)
4. Spike-phase locking (polar histograms + ISI)
5. Raster (L5 PN and PV)
6. Neuron positions (3D + 2D)
7. PV connectivity profiles (3D scatter + 2D projections)
8. Membrane potential traces
"""

import sys
import pickle
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D)

import numpy as np
from scipy.signal import welch, hilbert, butter, filtfilt
from scipy.ndimage import gaussian_filter


# ======================================================================
# Matplotlib style
# ======================================================================

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
})


# Population colours used consistently across figures.
COLORS = {
    'pn_l4': '#ED9703',
    'pn_l5': '#AF536F',
    'sst_l4': 'seagreen',
    'pv_l5': 'royalblue',
}


# ======================================================================
# Helpers
# ======================================================================

def _pop_rate(spiketimes, n_neurons, sim_time, bin_ms=1.0, smooth_ms=5.0):
    """Population mean firing rate in Hz, smoothed with a Gaussian."""
    bins = np.arange(0, sim_time + bin_ms, bin_ms)
    counts = np.zeros(len(bins) - 1)
    for i in range(n_neurons):
        c, _ = np.histogram(spiketimes[0][i], bins)
        counts += c
    rate = counts / (n_neurons * (bin_ms / 1000.0))
    if smooth_ms > 0:
        rate = gaussian_filter(rate, smooth_ms / bin_ms)
    t_centers = 0.5 * (bins[:-1] + bins[1:])
    return rate, t_centers


def _rate_heatmap(spiketimes, n_neurons, sim_time, bin_ms=100.0):
    """Per-neuron firing rate heatmap (Hz), sorted by mean rate."""
    bins = np.arange(0, sim_time + bin_ms, bin_ms)
    t_centers = 0.5 * (bins[:-1] + bins[1:])
    matrix = np.zeros((n_neurons, len(bins) - 1))
    for i in range(n_neurons):
        c, _ = np.histogram(spiketimes[0][i], bins)
        matrix[i] = c / (bin_ms / 1000.0)
    order = np.argsort(matrix.mean(axis=1))
    return matrix[order], t_centers


def _mean_neuron_rate(spiketimes, n_neurons, sim_time):
    """Mean individual neuron firing rate (Hz)."""
    duration_s = sim_time / 1000.0
    total_spikes = sum(len(spiketimes[0][i]) for i in range(n_neurons))
    return total_spikes / (n_neurons * duration_s)


def _peak_frequency(rate_signal, fs=1000.0, f_min=4.0, f_max=150.0):
    """Dominant oscillation frequency via Welch PSD."""
    if len(rate_signal) < 2:
        return float('nan')
    freqs, power = welch(
        rate_signal, fs=fs, nperseg=min(2048, len(rate_signal)))
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        return float('nan')
    return float(freqs[mask][np.argmax(power[mask])])


def _save(fig, out_dir, fname, save_as_svg=False):
    """Save a figure under `out_dir`."""
    ext = 'svg' if save_as_svg else 'png'
    fig.savefig(
        Path(out_dir) / f'{fname}.{ext}',
        bbox_inches='tight', dpi=150)
    plt.close(fig)


# ======================================================================
# Figure: per-population rate + heatmap
# ======================================================================

def plot_pop_rate_heatmap(spiketimes, n_neurons, sim_time,
                          label, color, fname, out_dir, save_as_svg):
    """Two-panel figure: population mean rate (top) + per-neuron heatmap."""
    rate, t_rate = _pop_rate(spiketimes, n_neurons, sim_time)
    hmap, t_hmap = _rate_heatmap(spiketimes, n_neurons, sim_time)
    mean_rate = float(rate.mean()) if rate.size > 0 else 0.0
    max_rate = float(rate.max()) if rate.size > 0 else 0.0

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={'height_ratios': [1, 2]})

    axes[0].plot(t_rate, rate, color=color, linewidth=1)
    axes[0].set_ylabel('Rate (Hz)')
    axes[0].set_title(
        f'{label} - mean {mean_rate:.1f} Hz, peak {max_rate:.1f} Hz')
    axes[0].set_xlim(t_rate[0], t_rate[-1])

    im = axes[1].imshow(
        hmap, aspect='auto',
        extent=[t_hmap[0], t_hmap[-1], 0, hmap.shape[0]],
        origin='lower', cmap='viridis')
    axes[1].set_ylabel('Neuron no. (sorted by rate)')
    axes[1].set_xlabel('Time (ms)')
    fig.colorbar(im, ax=axes[1], label='Rate (Hz)')

    fig.text(
        0.98, 0.97,
        f'n = {n_neurons}, mean {mean_rate:.2f} Hz, '
        f'peak {max_rate:.2f} Hz',
        ha='right', va='top', fontsize=10.8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    _save(fig, out_dir, fname, save_as_svg)


# ======================================================================
# Figure: stacked population oscillations + summary table
# ======================================================================

def plot_population_oscillations(rates, labels, colors, t_rate,
                                 out_dir, save_as_svg):
    """Stacked rate traces with peak-frequency annotations."""
    fig, axes = plt.subplots(len(rates), 1, figsize=(16, 12), sharex=True)
    for ax, rate, label, color in zip(axes, rates, labels, colors):
        peak_f = _peak_frequency(rate)
        ax.plot(t_rate, rate, color=color, linewidth=0.9)
        ax.set_ylabel('Rate (Hz)')
        ax.set_title(f'{label} - Peak oscillation freq: {peak_f:.1f} Hz')
    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('Population oscillations', y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, 'population_oscillations', save_as_svg)


def print_population_summary(rates, mfrs, labels):
    """Print mean rate and PSD peak per population."""
    print('\n-- Population statistics --------------------------------------')
    print(f'  {"Population":<14}  {"mean neuron rate":>17}  {"osc. peak":>10}')
    print(f'  {"-"*14}  {"-"*17}  {"-"*10}')
    for rate, mfr, label in zip(rates, mfrs, labels):
        peak_f = _peak_frequency(rate)
        peak_str = f'{peak_f:>7.1f} Hz' if not np.isnan(peak_f) else 'N/A'
        print(f'  {label:<14}  {mfr:>14.2f} Hz  {peak_str:>10}')
    print('-' * 55)


# ======================================================================
# Figure: PV L5 and PN L5 power spectra side by side
# ======================================================================

def plot_power_spectra(rate_pv, rate_pn, out_dir, save_as_svg):
    """Welch PSD for PV L5 and PN L5 with band shading."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
    for ax, signal, label in zip(
            axes, [rate_pv, rate_pn], ['PV L5', 'PN L5']):
        freqs, power = welch(
            signal, fs=1000.0, nperseg=min(1024, len(signal)))
        ax.plot(freqs, power, color='black')
        ax.axvspan(13, 30, alpha=0.3, color='pink')
        ax.axvspan(30, 80, alpha=0.1, color='green')
        ax.set_xlim(0, 120)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(f'{label} power spectrum')

    axes[1].legend(
        handles=[
            Patch(facecolor='pink', alpha=0.3, label='beta band'),
            Patch(facecolor='green', alpha=0.1, label='gamma band'),
        ],
        loc='upper right')
    fig.tight_layout()
    _save(fig, out_dir, 'power_spectra_pv_pn', save_as_svg)


# ======================================================================
# Figure: spike-phase locking (PN L5 and PV L5 vs gamma)
# ======================================================================

def plot_spike_phase_locking(rate_pv, t_rate,
                             spiketimes_pn, n_pn,
                             spiketimes_pv, n_pv,
                             out_dir, save_as_svg):
    """Polar phase histograms and ISI distributions vs PV-derived gamma."""

    def _bandpass_gamma(sig, lowcut=30.0, highcut=80.0,
                        fs=1000.0, order=4):
        nyq = fs / 2.0
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return filtfilt(b, a, sig)

    pv_gamma = _bandpass_gamma(rate_pv)
    pv_phase = np.angle(hilbert(pv_gamma))

    freqs_g, psd_g = welch(
        rate_pv, fs=1000.0, nperseg=min(2048, len(rate_pv)))
    mask_g = (freqs_g >= 20.0) & (freqs_g <= 100.0)
    peak_gamma_hz = (
        freqs_g[mask_g][np.argmax(psd_g[mask_g])]
        if mask_g.any() else 40.0)
    gamma_period = 1000.0 / peak_gamma_hz

    t0 = t_rate[0]
    dt = t_rate[1] - t_rate[0]

    def _phases_for_spikes(spike_times):
        idxs = np.round((spike_times - t0) / dt).astype(int)
        valid = (idxs >= 0) & (idxs < len(pv_phase))
        return pv_phase[idxs[valid]]

    def _all_spikes(spiketimes_struct, n):
        parts = [spiketimes_struct[0][i] for i in range(n)
                 if len(spiketimes_struct[0][i]) > 0]
        return np.concatenate(parts) if parts else np.array([])

    def _all_isis(spiketimes_struct, n):
        parts = [np.diff(np.sort(spiketimes_struct[0][i]))
                 for i in range(n)
                 if len(spiketimes_struct[0][i]) > 1]
        return np.concatenate(parts) if parts else np.array([])

    pn_spikes = _all_spikes(spiketimes_pn, n_pn)
    pv_spikes = _all_spikes(spiketimes_pv, n_pv)
    pn_phases = _phases_for_spikes(pn_spikes)
    pv_phases = _phases_for_spikes(pv_spikes)

    pn_mvl = (
        float(np.abs(np.mean(np.exp(1j * pn_phases))))
        if len(pn_phases) > 0 else 0.0)
    pv_mvl = (
        float(np.abs(np.mean(np.exp(1j * pv_phases))))
        if len(pv_phases) > 0 else 0.0)

    pn_isis = _all_isis(spiketimes_pn, n_pn)
    pv_isis = _all_isis(spiketimes_pv, n_pv)

    fig = plt.figure(figsize=(14, 10))
    ax_pn_pol = fig.add_subplot(2, 2, 1, projection='polar')
    ax_pv_pol = fig.add_subplot(2, 2, 2, projection='polar')
    ax_pn_isi = fig.add_subplot(2, 2, 3)
    ax_pv_isi = fig.add_subplot(2, 2, 4)

    n_bins_pol = 36
    bins_pol = np.linspace(-np.pi, np.pi, n_bins_pol + 1)
    bin_width = bins_pol[1] - bins_pol[0]

    for ax_pol, phases, mvl, label, color in [
        (ax_pn_pol, pn_phases, pn_mvl, 'PN L5', COLORS['pn_l5']),
        (ax_pv_pol, pv_phases, pv_mvl, 'PV L5', COLORS['pv_l5']),
    ]:
        counts, _ = np.histogram(phases, bins=bins_pol)
        counts = counts / counts.sum() if counts.sum() > 0 else counts
        ax_pol.bar(
            bins_pol[:-1], counts, width=bin_width, align='edge',
            color=color, alpha=0.75, edgecolor='white', linewidth=0.4)
        ax_pol.set_title(
            f'{label}\nPhase Locking Value (PLV) = {mvl:.3f}', pad=12)
        ax_pol.set_theta_zero_location('N')
        ax_pol.set_theta_direction(-1)
        rticks = ax_pol.get_yticks()
        ax_pol.set_yticklabels(
            [f'{int(round(r * 100))}%' for r in rticks], fontsize=7)

    isi_max = min(500.0, gamma_period * 8)
    for ax_isi, isis, label, color in [
        (ax_pn_isi, pn_isis, 'PN L5', COLORS['pn_l5']),
        (ax_pv_isi, pv_isis, 'PV L5', COLORS['pv_l5']),
    ]:
        ax_isi.hist(
            isis[isis <= isi_max], bins=100, color=color,
            alpha=0.75, edgecolor='white', linewidth=0.3)
        for k in range(1, 9):
            ax_isi.axvline(
                gamma_period * k, color='gray',
                linestyle='--', linewidth=0.7, alpha=0.6)
        ax_isi.legend(
            handles=[Line2D(
                [0], [0], color='gray', linestyle='--', linewidth=0.7,
                label=f'multiples of gamma period '
                      f'({gamma_period:.0f} ms)')],
            loc='upper right')
        ax_isi.set_xlabel('ISI (ms)')
        ax_isi.set_ylabel('ISI count')
        ax_isi.set_title(f'{label} Interspike Interval (ISI) distribution')

    fig.suptitle(
        f'Spike-phase locking to PV gamma ({peak_gamma_hz:.1f} Hz)\n'
        f'Dashed lines = multiples of gamma period '
        f'({gamma_period:.1f} ms)',
        y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, 'spike_phase_locking', save_as_svg)
    print(f'  Spike-phase locking (PLV): PN={pn_mvl:.3f}, PV={pv_mvl:.3f}')


# ======================================================================
# Figure: raster plot
# ======================================================================

def plot_raster_pv_pn(spiketimes_pv, n_pv, spiketimes_pn, n_pn,
                      out_dir, save_as_svg):
    """Raster plot with PV L5 (bottom) and PN L5 (top)."""
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_title('Raster Plot - PV L5 (steel blue) and PN L5 (pink)')

    for i, t_sp in enumerate(spiketimes_pv[0]):
        if len(t_sp) > 0:
            ax.plot(t_sp, np.full_like(t_sp, i, dtype=float),
                    '.', color='steelblue', markersize=1.5, alpha=0.6)

    for i in range(n_pn):
        t_sp = spiketimes_pn[0][i]
        if len(t_sp) > 0:
            ax.plot(t_sp, np.full_like(t_sp, n_pv + i, dtype=float),
                    '.', color='#CC618F', markersize=2, alpha=0.6)

    ax.axhline(n_pv, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.legend(handles=[
        Patch(facecolor='#CC618F', alpha=0.8, label='PN L5'),
        Patch(facecolor='steelblue', alpha=0.8, label='PV L5'),
        mlines.Line2D([], [], color='k', linewidth=0.8, linestyle='--',
                      label='PV / PN L5 boundary'),
    ], fontsize=12.0, loc='upper right')

    fig.tight_layout()
    _save(fig, out_dir, 'raster_pv_pn_l5', save_as_svg)


# ======================================================================
# Figure: 3D and 2D neuron position plots
# ======================================================================

def plot_neuron_positions(positions, spatial_radius, l5_z_min,
                          out_dir, save_as_svg):
    """3D and 2D views of neuron positions in the cortical column."""
    UM_SCALE = 200.0
    r_um = spatial_radius * UM_SCALE

    pn_l4 = np.asarray(positions['pn_l4']) * UM_SCALE
    sst_l4 = np.asarray(positions['sst_l4']) * UM_SCALE
    pn_l5 = np.asarray(positions['pn_l5']) * UM_SCALE
    pv_l5 = np.asarray(positions['pv_l5']) * UM_SCALE

    pop_specs = [
        (pn_l4, 'orange', 'PN L4'),
        (sst_l4, 'seagreen', 'SST L4'),
        (pn_l5, '#BF5985', 'PN L5'),
        (pv_l5, 'steelblue', 'PV'),
    ]

    # 3D plot
    z_shift = abs(l5_z_min) * UM_SCALE  # moves L5 floor to z=0

    fig_3d = plt.figure(figsize=(14, 12))
    ax = fig_3d.add_subplot(111, projection='3d')
    ax.set_position([0.05, 0.05, 0.82, 0.90])

    for pos, color, label in pop_specs:
        ax.scatter(pos[:, 0] + r_um, pos[:, 1] + r_um, pos[:, 2] + z_shift,
                   color=color, label=label, alpha=0.7, marker='o', s=35)

    grid_xy = np.linspace(0, 2 * r_um, 30)
    gx, gy = np.meshgrid(grid_xy, grid_xy)
    gz = np.full_like(gx, z_shift)
    ax.plot_surface(gx, gy, gz, alpha=0.12, color='grey', linewidth=0)

    ax.set_xlim(0, 2 * r_um); ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_ylim(0, 2 * r_um); ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_zlim(0, 300); ax.set_zticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_box_aspect([1, 1, 1.5])
    ax.set_xlabel('(um)', labelpad=12)
    ax.set_ylabel('Width (um)', labelpad=12)
    ax.set_zlabel('Depth (um)', labelpad=12)
    ax.legend(loc='upper left', fontsize=14.4)
    fig_3d.tight_layout()
    _save(fig_3d, out_dir, 'neuron_positions', save_as_svg)

    # 2D overview
    fig2, (ax_top, ax_side) = plt.subplots(
        1, 2, figsize=(15, 10),
        gridspec_kw={'width_ratios': [1.05, 0.95]})

    for pos, color, _ in pop_specs:
        ax_top.scatter(pos[:, 0] + r_um, pos[:, 1] + r_um,
                       s=28, c=color, alpha=0.75, edgecolors='none')
    ax_top.set_aspect('equal', adjustable='box')
    ax_top.set_xlim(0, 2 * r_um)
    ax_top.set_ylim(0, 2 * r_um)
    ax_top.set_xlabel('X (um)')
    ax_top.set_ylabel('Y (um)')
    ax_top.set_title('Top view')
                              
    all_pos = np.vstack([pn_l4, sst_l4, pn_l5, pv_l5])
    zmin, zmax = float(np.min(all_pos[:, 2])), float(np.max(all_pos[:, 2]))
    z_offset = -zmin

    z_l4 = np.concatenate([pn_l4[:, 2], sst_l4[:, 2]])
    z_l5 = np.concatenate([pn_l5[:, 2], pv_l5[:, 2]])
    boundary_z = 0.5 * (np.min(z_l4) + np.max(z_l5))
    boundary_z_sh = boundary_z + z_offset
    zmax_sh = zmax + z_offset

    for pos, color, _ in pop_specs:
        ax_side.scatter(pos[:, 0] + r_um, pos[:, 2] + z_offset,
                        s=28, c=color, alpha=0.75, edgecolors='none')

    ax_side.axhline(
        boundary_z_sh, color='0.35', linestyle='--', linewidth=1.2)
    ax_side.text(
        2 * r_um * 1.04, 0.5 * (boundary_z_sh + zmax_sh), 'L4',
        color='0.25', va='center', fontsize=14.4,
        fontweight='bold', clip_on=False)
    ax_side.text(
        2 * r_um * 1.04, 0.5 * boundary_z_sh, 'L5',
        color='0.25', va='center', fontsize=14.4,
        fontweight='bold', clip_on=False)

    ax_side.set_xlim(0, 2 * r_um)
    ax_side.set_xticks([0, 100, 200])
    ax_side.set_ylim(0, zmax_sh)
    ax_side.set_xlabel('X (um)')
    ax_side.set_ylabel('Depth (um)')
    ax_side.set_title('Side view')

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor=color, markersize=12)
        for _, color, label in pop_specs
    ]
    ax_side.legend(
        handles=legend_handles, loc='upper right',
        frameon=True, fontsize=16.8)

    fig2.tight_layout()
    _save(fig2, out_dir, 'neuron_positions_2d', save_as_svg)


# ======================================================================
# Figure: PV connectivity profiles
# ======================================================================

def plot_pv_profiles(profiles, out_dir, save_as_svg):
    """3D scatter and 2D projections for PV connectivity profiles."""
    if not profiles:
        return

    uni = np.array([p['n_uni_chem'] for p in profiles.values()])
    bidi = np.array([p['n_bidi_chem'] for p in profiles.values()])
    gap = np.array([p['n_gap'] for p in profiles.values()])
    total = np.array([p['total_connected'] for p in profiles.values()])

    norm = Normalize(vmin=total.min(), vmax=total.max())
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'royalblue', ['#98B0E5', '#011979'])

    # 3D scatter
    fig_3d = plt.figure(figsize=(12, 10))
    ax3 = fig_3d.add_subplot(111, projection='3d')
    sc = ax3.scatter(
        uni, bidi, gap, c=total, cmap=cmap, norm=norm,
        s=50, alpha=0.75, edgecolors='none')
    ax3.set_xlabel('Unidirectional chemical', fontsize=14.4, labelpad=12)
    ax3.set_ylabel('Reciprocal chemical', fontsize=14.4, labelpad=12)
    ax3.set_zlabel('Electrical (gap junctions)', fontsize=14.4, labelpad=10)
    ax3.set_title('PV neuron connectivity profiles',
                  fontsize=16.8, pad=20)
    ax3.view_init(elev=25, azim=-60)
    ax3.invert_xaxis()
    ax3.invert_yaxis()
    fig_3d.colorbar(sc, ax=ax3, label='Total connected partners',
                    shrink=0.6, pad=0.1)
    fig_3d.tight_layout()
    _save(fig_3d, out_dir, 'pv_profiles_3d', save_as_svg)

    # 2D projections
    fig_proj, axes_p = plt.subplots(2, 2, figsize=(14, 12))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    pt_colors = cmap(norm(total))
    ec = pt_colors.copy()
    ec[:, 3] = 1.0

    panel_specs = [
        (axes_p[0, 0], bidi, gap,
         'Reciprocal chem partners', 'Gap junction partners',
         'Reciprocal chem vs gap junctions'),
        (axes_p[0, 1], uni, gap,
         'Unidirectional chem partners', 'Gap junction partners',
         'Unidirectional chem vs gap junctions'),
        (axes_p[1, 0], uni, bidi,
         'Unidirectional chem partners', 'Reciprocal chem partners',
         'Unidirectional vs reciprocal chem'),
    ]

    correlations = {}
    for ax, x, y, xlabel, ylabel, title in panel_specs:
        ax.scatter(x, y, c=total, cmap=cmap, norm=norm,
                   s=45, alpha=0.7, edgecolors=ec, linewidths=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        r = np.corrcoef(x, y)[0, 1]
        correlations[title] = r
        ax.text(
            0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
            va='top', fontsize=13.2,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Histogram of total connectivity
    counts, bin_edges, patches = axes_p[1, 1].hist(
        total, bins=20, edgecolor='white', alpha=0.85)
    bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for patch, mid in zip(patches, bin_mids):
        fc = cmap(norm(mid))
        patch.set_facecolor(fc)
        patch.set_edgecolor((*fc[:3], 1.0))
        patch.set_linewidth(0.5)
    fig_proj.colorbar(
        sm, ax=axes_p[1, 1], label='Total connected partners')
    pctl_90 = np.percentile(total, 90)
    axes_p[1, 1].axvline(
        pctl_90, color='crimson', linestyle='--', linewidth=1.5,
        label=f'90th pctl ({pctl_90:.0f})')
    axes_p[1, 1].set_xlabel('Total connected partners')
    axes_p[1, 1].set_ylabel('Count')
    axes_p[1, 1].set_title('Hub distribution')
    axes_p[1, 1].legend(fontsize=12.0)

    fig_proj.suptitle(
        'PV connectivity profiles - 2D projections',
        fontsize=16.8, y=1.01)
    fig_proj.tight_layout()
    _save(fig_proj, out_dir, 'pv_profiles_2d', save_as_svg)

    print('\n-- PV connectivity profile correlations -----------------------')
    for title, r in correlations.items():
        print(f'  {title}: r = {r:.3f}')
    print(f'  Total connected: {total.mean():.1f} +/- {total.std():.1f}')
    print(f'  Hub threshold (90th pctl): {pctl_90:.0f} partners')
    print('-' * 55)


# ======================================================================
# Figure: membrane potential traces
# ======================================================================

def plot_membrane_potentials(sim_data, out_dir, save_as_svg,
                             zoom_start=100, zoom_end=3100):
    """V_m traces with synthetic spike waveforms inserted."""

    def _insert_spikes(t_array, vm_array, spike_times, v_peak=20.0):
        vm_out = vm_array.copy()
        for t_sp in spike_times:
            idx = np.searchsorted(t_array, t_sp)
            if 0 < idx < len(vm_out) - 1:
                vm_out[idx] = v_peak
                vm_out[idx - 1] = (vm_array[idx - 1] + v_peak) / 2
        return vm_out

    mm_idx = sim_data['mm_neuron_idx']

    pops = [
        (sim_data['vm_pn_l4'], sim_data['t_vm_pn_l4'],
         sim_data['spiketimes_pn_l4'], 'PN L4', COLORS['pn_l4']),
        (sim_data['vm_pn_l5'], sim_data['t_vm_pn_l5'],
         sim_data['spiketimes_pn_l5'], 'PN L5', COLORS['pn_l5']),
        (sim_data['vm_sst_l4'], sim_data['t_vm_sst_l4'],
         sim_data['spiketimes_sst_l4'], 'SST L4', COLORS['sst_l4']),
        (sim_data['vm_pv_l5'], sim_data['t_vm_pv_l5'],
         sim_data['spiketimes_pv_l5'], 'PV L5', COLORS['pv_l5']),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    for ax, (vm, t_vm, spiketimes, label, color) in zip(axes, pops):
        if len(vm) == 0:
            ax.set_title(f'{label} neuron {mm_idx} - no Vm data recorded')
            continue
        sp_times = spiketimes[0][mm_idx]
        vm_with_spikes = _insert_spikes(t_vm, vm, sp_times)
        ax.plot(t_vm, vm_with_spikes, color=color, linewidth=0.9)
        ax.set_xlim(zoom_start, zoom_end)
        ax.set_ylabel('Vm (mV)')
        ax.set_title(label)

    axes[-1].set_xlabel('Time (ms)')
    fig.tight_layout()
    _save(fig, out_dir, 'membrane_potentials', save_as_svg)


# ======================================================================
# Main entry point
# ======================================================================

def main(sim_data_path):
    with open(sim_data_path, 'rb') as f:
        sim_data = pickle.load(f)

    out_dir = Path(sim_data_path).parent / 'Figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    save_as_svg = False  # Toggle to write SVGs instead of PNGs

    sim_time = sim_data['sim_time']
    n_pn_l4 = sim_data['pn_l4_count']
    n_pn_l5 = sim_data['pn_l5_count']
    n_sst_l4 = sim_data['sst_l4_count']
    n_pv_l5 = sim_data['pv_l5_count']

    spiketimes_pn_l4 = sim_data['spiketimes_pn_l4']
    spiketimes_pn_l5 = sim_data['spiketimes_pn_l5']
    spiketimes_sst_l4 = sim_data['spiketimes_sst_l4']
    spiketimes_pv_l5 = sim_data['spiketimes_pv_l5']

    print(f'Plotting from {sim_data_path}')
    print(f'Output directory: {out_dir}')

    # Per-population rate + heatmap figures
    plot_pop_rate_heatmap(spiketimes_pn_l4, n_pn_l4, sim_time,
                          'PN L4', 'orange',
                          'pop_rate_heatmap_pn_l4', out_dir, save_as_svg)
    plot_pop_rate_heatmap(spiketimes_pn_l5, n_pn_l5, sim_time,
                          'PN L5', 'royalblue',
                          'pop_rate_heatmap_pn_l5', out_dir, save_as_svg)
    plot_pop_rate_heatmap(spiketimes_sst_l4, n_sst_l4, sim_time,
                          'SST L4', 'seagreen',
                          'pop_rate_heatmap_sst_l4', out_dir, save_as_svg)
    plot_pop_rate_heatmap(spiketimes_pv_l5, n_pv_l5, sim_time,
                          'PV L5', '#CC618F',
                          'pop_rate_heatmap_pv_l5', out_dir, save_as_svg)

    # Compute per-population rates once
    rate_pn_l4, t_rate = _pop_rate(spiketimes_pn_l4, n_pn_l4, sim_time)
    rate_pn_l5, _ = _pop_rate(spiketimes_pn_l5, n_pn_l5, sim_time)
    rate_sst_l4, _ = _pop_rate(spiketimes_sst_l4, n_sst_l4, sim_time)
    rate_pv_l5, _ = _pop_rate(spiketimes_pv_l5, n_pv_l5, sim_time)

    mfr_pn_l4 = _mean_neuron_rate(spiketimes_pn_l4, n_pn_l4, sim_time)
    mfr_pn_l5 = _mean_neuron_rate(spiketimes_pn_l5, n_pn_l5, sim_time)
    mfr_sst_l4 = _mean_neuron_rate(spiketimes_sst_l4, n_sst_l4, sim_time)
    mfr_pv_l5 = _mean_neuron_rate(spiketimes_pv_l5, n_pv_l5, sim_time)

    print_population_summary(
        [rate_pn_l4, rate_pn_l5, rate_sst_l4, rate_pv_l5],
        [mfr_pn_l4, mfr_pn_l5, mfr_sst_l4, mfr_pv_l5],
        ['PN L4', 'PN L5', 'SST L4', 'PV L5'])

    plot_population_oscillations(
        rates=[rate_pn_l4, rate_pn_l5, rate_sst_l4, rate_pv_l5],
        labels=['PN L4', 'PN L5', 'SST L4', 'PV L5'],
        colors=[COLORS['pn_l4'], COLORS['pn_l5'],
                COLORS['sst_l4'], COLORS['pv_l5']],
        t_rate=t_rate, out_dir=out_dir, save_as_svg=save_as_svg)

    plot_power_spectra(rate_pv_l5, rate_pn_l5, out_dir, save_as_svg)

    plot_spike_phase_locking(
        rate_pv_l5, t_rate,
        spiketimes_pn_l5, n_pn_l5,
        spiketimes_pv_l5, n_pv_l5,
        out_dir, save_as_svg)

    plot_raster_pv_pn(
        spiketimes_pv_l5, n_pv_l5,
        spiketimes_pn_l5, n_pn_l5,
        out_dir, save_as_svg)

    # spatial_radius and l5_z_min are config values that aren't pickled.
    # Standard values from the YAML are 0.5 and -1.0.
    plot_neuron_positions(
        positions=sim_data['positions'],
        spatial_radius=0.5, l5_z_min=-1.0,
        out_dir=out_dir, save_as_svg=save_as_svg)

    plot_pv_profiles(
        sim_data['pv_profiles'], out_dir, save_as_svg)

    plot_membrane_potentials(sim_data, out_dir, save_as_svg)

    print(f'\nAll figures saved to {out_dir}')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Default: plot the most recent simulation in saved_simulations/
        latest_dirs = sorted(
            Path('saved_simulations').glob('*/'),
            key=lambda p: p.stat().st_mtime,
            reverse=True)
        if not latest_dirs:
            print('No simulations found in saved_simulations/.')
            print('Run `python -m model.run` first, or pass a path:')
            print('  python -m plotting.plot path/to/sim_data.pkl')
            sys.exit(1)
        sim_data_path = latest_dirs[0] / 'sim_data.pkl'
        print(f'Auto-selected most recent run: {sim_data_path}')
    elif len(sys.argv) == 2:
        sim_data_path = sys.argv[1]
    else:
        print('Usage:')
        print('  python -m plotting.plot                         '
              '# plot most recent run')
        print('  python -m plotting.plot path/to/sim_data.pkl    '
              '# plot specific run')
        sys.exit(1)

    main(sim_data_path)
