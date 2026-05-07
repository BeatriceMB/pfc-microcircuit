#!/usr/bin/env python3
"""
Condition-specific plotting for a single PV perturbation experiment run.

Usage:
    python3 -m experiments.plot_pv_experiment [npz_file]

If an .npz file is provided, the script plots that single run.
If no argument is provided, it automatically finds and plots all files
matching pv_plot_data_*.npz in ./pv_plot_data/ (or the current
directory as a fallback).
"""

import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, correlate, correlation_lags
from matplotlib.ticker import MultipleLocator

#PN_COLOR = "#6D6C79" 
#PV_COLOR = "#17171C" 


PN_COLOR = "#BE5A78"
PV_COLOR = "#38486B"

#PN_COLOR = "#AE5579" 
#PV_COLOR = "#007A90" 



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
    'font.size': 8.0,
    'axes.labelsize': 9.0,
    'axes.titlesize': 9.0,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
})


def _condition_annotation(condition, level):
    if condition in ('peripheral_first', 'random_order', 'hub_first'):
        return r'$I_{e,\mathrm{PV}}=0$'
    if condition in ('gap_recip_chem', 'uni_chem_only', 'global_gap'):
        if abs(level - 1.0) < 1e-9:
            return r'$w_{\mathrm{PV\mathrm{-}PV}}=0$'
        return rf'$w_{{\mathrm{{PV\mathrm{{-}}PV}}}}={int(round((1.0-level)*100))}\%$'
    return None


def _panel_label(ax, condition, level):
    ann = _condition_annotation(condition, level)
    if ann is None:
        return
    ax.text(0.98, 0.94, ann, transform=ax.transAxes, ha='right', va='top', fontsize=7.5)


def plot_one(npz_file):
    d = np.load(npz_file, allow_pickle=True)

    SEED = int(d['seed'])
    CONDITION = str(d['condition'])
    LEVEL = float(d['level'])
    nn_sim_time = float(d['nn_sim_time'])
    nn_total_pv = int(d['nn_total_pv'])
    nn_pn_exc_l5_count = int(d['nn_pn_exc_l5_count'])

    rate_pn = d['rate_pn']
    t_pn = d['t_pn']
    rate_pv = d['rate_pv']
    t_pv = d['t_pv']
    coh_freqs = d['coh_freqs']
    coh_spectrum = d['coh_spectrum']
    mfr_pn = float(d['mfr_pn'])
    mfr_pv = float(d['mfr_pv'])
    peak_freq_pn = float(d['peak_freq_pn'])
    peak_freq_pv = float(d['peak_freq_pv'])

    spiketimes_pv_l5 = d['spiketimes_pv_l5']
    spiketimes_pn_l5 = d['spiketimes_pn_l5']

    fig_dir = pathlib.Path(f"pv_figures/{SEED}/{CONDITION}_{LEVEL:.1f}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    np.savez(fig_dir / 'coherence_spectrum.npz',
             freqs=coh_freqs, coherence=coh_spectrum,
             seed=SEED, condition=CONDITION, level=LEVEL)

    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    ax.plot(coh_freqs, coh_spectrum, color='k', linewidth=0.8)
    ax.axvspan(30, 80, alpha=0.08, color='green', label='gamma band')
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Coherence')
    
    _panel_label(ax, CONDITION, LEVEL)
    ax.legend()
    fig.subplots_adjust(top=0.94)
    fig.savefig(fig_dir / 'coherence_spectrum.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    for ax, sig, label, color in [
        (axes[0], rate_pn, 'PN L5', 'black'),
        (axes[1], rate_pv, 'PV L5', 'black'),
    ]:
        freqs, psd = welch(sig, fs=1000.0, nperseg=min(1024, len(sig)))
        ax.plot(freqs, psd, color=color, linewidth=0.8)
        ax.axvspan(13, 30, alpha=0.12, color='#FFB3C6', label='beta')
        ax.axvspan(30, 80, alpha=0.08, color='green', label='gamma')
        ax.set_xlim(0, 150)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        
        _panel_label(ax, CONDITION, LEVEL)
        ax.legend()
    fig.subplots_adjust(top=0.93, wspace=0.30)
    fig.savefig(fig_dir / 'power_spectra.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    t_window = max(0, nn_sim_time - 500)
    n_pv_plot = nn_total_pv
    for i in range(n_pv_plot):
        sp = spiketimes_pv_l5[i]
        sp = sp[sp >= t_window]
        if len(sp) > 0:
            ax.plot(sp, np.full_like(sp, i, dtype=float), '.', color=PV_COLOR, markersize=1, alpha=0.9)
    for i in range(nn_pn_exc_l5_count):
        sp = spiketimes_pn_l5[i]
        sp = sp[sp >= t_window]
        if len(sp) > 0:
            ax.plot(sp, np.full_like(sp, n_pv_plot + i, dtype=float), '.', color=PN_COLOR, markersize=1, alpha=0.9)
    ax.axhline(n_pv_plot, color='k', linewidth=0.8, linestyle='--', label='PV / PN boundary')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Number of neurons')

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='.', color='w', markerfacecolor=PV_COLOR, markersize=6, label='PV'),
        Line2D([0], [0], marker='.', color='w', markerfacecolor=PN_COLOR, markersize=6, label='PN'),
        Line2D([0], [0], color='k', linewidth=0.8, linestyle='--', label='PV / PN boundary'),
    ]
    ax.legend(handles=legend_handles, loc='upper left')

    _panel_label(ax, CONDITION, LEVEL)
    fig.subplots_adjust(top=0.94)
    fig.savefig(fig_dir / 'raster.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.6), sharex=True)
    axes[0].plot(t_pn, rate_pn, color=PN_COLOR, linewidth=0.4)
    axes[0].set_ylabel('PN L5 rate (Hz)')
    
    _panel_label(axes[0], CONDITION, LEVEL)
    axes[0].text(0.98, 0.92, f'mean {mfr_pn:.1f} Hz | peak freq {peak_freq_pn:.1f} Hz',
                 transform=axes[0].transAxes, ha='right', va='top', fontsize=6.8)
    axes[1].plot(t_pv, rate_pv, color=PV_COLOR, linewidth=0.4)
    axes[1].set_ylabel('PV L5 rate (Hz)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].text(0.98, 0.92, f'mean {mfr_pv:.1f} Hz | peak freq {peak_freq_pv:.1f} Hz',
                 transform=axes[1].transAxes, ha='right', va='top', fontsize=6.8)
    axes[1].set_xlim(0, nn_sim_time)
    axes[1].xaxis.set_major_locator(MultipleLocator(1000))
    fig.subplots_adjust(top=0.92, hspace=0.18)
    fig.savefig(fig_dir / 'rate_traces_full.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    t_zoom = max(0, nn_sim_time - 1000)
    mask_pn = t_pn >= t_zoom
    mask_pv = t_pv >= t_zoom

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.6), sharex=True)
    axes[0].plot(t_pn[mask_pn], rate_pn[mask_pn], color=PN_COLOR, linewidth=0.9)
    axes[0].set_ylabel('PN L5 rate (Hz)')
    
    _panel_label(axes[0], CONDITION, LEVEL)
    axes[0].text(0.98, 0.92, f'mean {mfr_pn:.1f} Hz | peak freq {peak_freq_pn:.1f} Hz',
                 transform=axes[0].transAxes, ha='right', va='top', fontsize=6.8)
    axes[1].plot(t_pv[mask_pv], rate_pv[mask_pv], color=PV_COLOR, linewidth=0.9)
    axes[1].set_ylabel('PV L5 rate (Hz)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].text(0.98, 0.92, f'mean {mfr_pv:.1f} Hz | peak freq {peak_freq_pv:.1f} Hz',
                 transform=axes[1].transAxes, ha='right', va='top', fontsize=6.8)
    fig.subplots_adjust(top=0.92, hspace=0.18)
    fig.savefig(fig_dir / 'rate_traces.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.8), gridspec_kw={'height_ratios': [2, 1]})
    pn_zoom = rate_pn[mask_pn]
    pv_zoom = rate_pv[mask_pv]
    t_zoom_pn = t_pn[mask_pn]
    t_zoom_pv = t_pv[mask_pv]
    min_len_plot = min(len(pn_zoom), len(pv_zoom), len(t_zoom_pn), len(t_zoom_pv))
    pn_zoom = pn_zoom[:min_len_plot]
    pv_zoom = pv_zoom[:min_len_plot]
    t_zoom_common = t_zoom_pn[:min_len_plot]
    pn_norm = pn_zoom / max(pn_zoom.max(), 1e-6)
    pv_norm = pv_zoom / max(pv_zoom.max(), 1e-6)
    axes[0].plot(t_zoom_common, pn_norm, color=PN_COLOR, linewidth=0.8, alpha=1.0, label='PN L5')
    axes[0].plot(t_zoom_common, pv_norm, color=PV_COLOR, linewidth=0.8, alpha=1.0, label='PV L5')
    axes[0].set_ylabel('Normalised rate')
    
    _panel_label(axes[0], CONDITION, LEVEL)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0.01, 0.88))
    pn_cc = pn_norm - np.mean(pn_norm)
    pv_cc = pv_norm - np.mean(pv_norm)
    xcorr = correlate(pn_cc, pv_cc, mode='full')
    lags = correlation_lags(len(pn_cc), len(pv_cc), mode='full')
    denom = np.sqrt(np.sum(pn_cc**2) * np.sum(pv_cc**2))
    if denom > 0:
        xcorr = xcorr / denom
    lag_ms = lags * 1.0
    axes[1].plot(lag_ms, xcorr, color='black', linewidth=0.8)
    axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('Lag (ms)')
    axes[1].set_ylabel('Cross-corr')
    
    fig.subplots_adjust(top=0.91, hspace=0.28)
    fig.savefig(fig_dir / 'rate_overlay.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}/")


def main():
    if len(sys.argv) == 2:
        plot_one(sys.argv[1])
        print("\n  Done.\n")
        return

    if len(sys.argv) == 1:
        # Look for npz files in the dedicated folder first, fall back to
        # current directory for backward compatibility with old runs.
        plot_data_dir = pathlib.Path('pv_plot_data')
        if plot_data_dir.exists():
            npz_files = sorted(plot_data_dir.glob('pv_plot_data_*.npz'))
        else:
            npz_files = sorted(pathlib.Path('.').glob('pv_plot_data_*.npz'))
        if not npz_files:
            print('No files matching pv_plot_data_*.npz found '
                  'in pv_plot_data/ or current directory.')
            sys.exit(1)
        print(f'Found {len(npz_files)} plot data files.')
        for npz_file in npz_files:
            print(f'Plotting {npz_file} ...')
            plot_one(str(npz_file))
        print("\n  Done.\n")
        return

    print(__doc__)
    sys.exit(1)


if __name__ == '__main__':
    main()
