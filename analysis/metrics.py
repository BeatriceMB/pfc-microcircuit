#!/usr/bin/env python
"""
Spectral and oscillatory metrics for population firing rate signals.

This module provides three groups of analysis functions:

1. Spectral helpers (`gamma_band_power`, `peak_frequency`,
   `gamma_coherence`) -- single-scalar summaries of population
   oscillatory state. Used both for baseline characterisation and for
   between-condition comparisons in perturbation experiments.

2. Burst-duration analysis (`compute_burst_metrics`) -- threshold
   crossings, mean burst duration, and frequency from upward
   crossings. Operates on a single population at a time.

3. Phase analysis (`compute_phase_metrics`) -- inter-population phase
   offsets and average peak heights for an excitatory-inhibitory
   pair. Used to characterise PING / ING regime transitions.

All functions operate on smoothed population rate signals (1-D numpy
arrays) sampled at `TIME_RESOLUTION` ms. Convert spike trains to rate
signals using `analysis.population.rate_code_spikes` before calling.
"""

import numpy as np
from scipy.signal import find_peaks, welch, coherence


# Sampling resolution of the rate signals (ms). Must match
# `parameters.time_resolution` to keep the time axis consistent.
TIME_RESOLUTION = 0.1


# ======================================================================
# Spectral metrics (single-scalar summaries)
# ======================================================================

def gamma_band_power(rate_signal, fs=1000.0, f_low=30.0, f_high=80.0):
    """
    Integrated power spectral density in the gamma band.

    Parameters
    ----------
    rate_signal : 1-D array
        Smoothed population firing rate (spikes/s).
    fs : float
        Sampling frequency in Hz (default 1000 Hz, matching
        TIME_RESOLUTION = 0.1 ms after appropriate downsampling).
    f_low, f_high : float
        Gamma band edges in Hz (default 30-80 Hz).

    Returns
    -------
    float
        Integrated PSD over [f_low, f_high]. Comparable across
        conditions provided the same band edges are used throughout.
    """
    freqs, psd = welch(
        rate_signal, fs=fs, nperseg=min(2048, len(rate_signal)))
    mask = (freqs >= f_low) & (freqs <= f_high)
    return float(np.trapz(psd[mask], freqs[mask]))


def peak_frequency(rate_signal, fs=1000.0, f_low=4.0, f_high=150.0):
    """
    Dominant oscillation frequency within a given band.

    Parameters
    ----------
    rate_signal : 1-D array
        Smoothed population firing rate.
    fs : float
        Sampling frequency in Hz.
    f_low, f_high : float
        Search band edges in Hz (default 4-150 Hz, covering theta
        through high gamma).

    Returns
    -------
    float
        Frequency (Hz) at which the PSD attains its maximum within
        the search band.
    """
    freqs, psd = welch(
        rate_signal, fs=fs, nperseg=min(2048, len(rate_signal)))
    mask = (freqs >= f_low) & (freqs <= f_high)
    return float(freqs[mask][np.argmax(psd[mask])])


def gamma_coherence(sig1, sig2, fs=1000.0, f_low=30.0, f_high=80.0):
    """
    Mean magnitude-squared coherence between two rate signals
    in the gamma band.

    Parameters
    ----------
    sig1, sig2 : 1-D array
        Population rate signals of equal length.
    fs : float
        Sampling frequency in Hz.
    f_low, f_high : float
        Gamma band edges in Hz.

    Returns
    -------
    float
        Mean magnitude-squared coherence over [f_low, f_high].
        Returns 0.0 if no frequency bins fall within the band.
    """
    freqs, coh = coherence(
        sig1, sig2, fs=fs, nperseg=min(1024, len(sig1)))
    mask = (freqs >= f_low) & (freqs <= f_high)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(coh[mask]))


# ======================================================================
# Burst-duration metrics (single population)
# ======================================================================

def compute_burst_metrics(rate_signal, threshold):
    """
    Burst duration, frequency, and variability for a single population.

    A burst is defined as the interval between an upward crossing
    (signal rising through `threshold`) and the following downward
    crossing. The frequency is computed from the inter-burst interval
    (time between successive upward crossings).

    Parameters
    ----------
    rate_signal : 1-D array
        Smoothed population rate signal.
    threshold : float
        Crossing threshold in the same units as `rate_signal`.

    Returns
    -------
    metrics : dict
        Keys:
            'mean_burst_duration_ms' : float
                Mean duration of a burst (ms).
            'burst_duration_variance' : float
                Variance of burst durations (ms^2).
            'burst_duration_cv' : float
                Coefficient of variation (%, 100 * std / mean).
            'frequency_hz' : float
                Oscillation frequency (Hz) from inter-burst intervals.
            'upward_crossings' : list of int
                Sample indices where the signal rose through threshold.
            'downward_crossings' : list of int
                Sample indices where the signal fell back through threshold.
    """
    upward_indices, downward_indices = _find_threshold_crossings(
        rate_signal, threshold)

    # Discard any incomplete first burst and align list lengths
    min_length = min(len(downward_indices), len(upward_indices))
    aligned_down = downward_indices[1:min_length]
    aligned_up = upward_indices[1:min_length]

    burst_durations = (
        np.subtract(aligned_down, aligned_up) * TIME_RESOLUTION)
    bd_mean = float(np.mean(burst_durations))
    bd_variance = float(np.var(burst_durations))
    bd_cv = (np.std(burst_durations) / bd_mean) * 100 if bd_mean > 0 else 0.0

    return {
        'mean_burst_duration_ms': round(bd_mean, 2),
        'burst_duration_variance': round(bd_variance, 2),
        'burst_duration_cv': round(bd_cv, 2),
        'frequency_hz': round(_crossings_to_frequency(aligned_up), 2),
        'upward_crossings': aligned_up,
        'downward_crossings': aligned_down,
    }


def _find_threshold_crossings(rate_signal, threshold):
    """
    Detect upward and downward threshold crossings in a rate signal.

    Returns
    -------
    upward_indices, downward_indices : list of int
        Sample indices of the crossings.
    """
    upward_indices = []
    downward_indices = []
    crossing = False

    for index, value in enumerate(rate_signal):
        if value >= threshold and not crossing:
            upward_indices.append(index)
            crossing = True
        elif value < threshold and crossing:
            downward_indices.append(index)
            crossing = False

    return upward_indices, downward_indices


def _crossings_to_frequency(crossings):
    """
    Convert a sequence of crossing indices to a frequency (Hz).

    Parameters
    ----------
    crossings : sequence of int
        Sample indices of successive same-direction crossings.

    Returns
    -------
    float
        Oscillation frequency, computed as
        1000 / mean inter-crossing interval (ms). Returns 0.0 if
        fewer than two crossings are provided.
    """
    if len(crossings) < 2:
        return 0.0
    period_ms = np.mean(np.diff(crossings)) * TIME_RESOLUTION
    return float(1000 / period_ms)


# ======================================================================
# Phase metrics (excitatory-inhibitory population pair)
# ======================================================================

def compute_phase_metrics(exc_rate, inh_rate,
                          exc_rate_raw, inh_rate_raw,
                          pop_label, peak_threshold,
                          min_peak_dist=1000):
    """
    Inter-population phase offset, frequency, and peak heights.

    Detects peaks in both population rate signals, extracts strictly
    alternating excitatory-inhibitory peaks, and computes the phase
    offset between consecutive cross-population peaks normalised by
    the average period.

    Parameters
    ----------
    exc_rate, inh_rate : 1-D array
        Smoothed excitatory and inhibitory population rate signals.
    exc_rate_raw, inh_rate_raw : 1-D array
        Unsmoothed (raw) rate signals, used for peak height estimates.
    pop_label : str
        Label used in printed output (e.g. 'L5', 'L4').
    peak_threshold : float
        Fraction-of-peak threshold for the underlying `find_peaks`
        call. Also used as the minimum-height threshold for the
        peak-height calculation.
    min_peak_dist : int, optional
        Minimum sample distance between consecutive peaks
        (default 1000).

    Returns
    -------
    metrics : dict or None
        If successful, dict with keys:
            'mean_phase_deg' : float
                Inter-population phase offset (degrees, 0-180).
                Wrapped to keep symmetric: phases > 180 are reflected.
            'phase_variance' : float
            'phase_cv' : float
                Coefficient of variation (%) of phase across pairs.
            'mean_frequency_hz' : float
                Average oscillation frequency.
            'mean_peak_height_exc' : float
            'mean_peak_height_inh' : float
                Average peak heights of the raw rate signals.
            'alternating_exc_peaks' : list of int
            'alternating_inh_peaks' : list of int
                Sample indices of the alternating peaks.
        None if peak detection fails.
    """
    try:
        (mean_phase, phase_variance, phase_cv, mean_freq,
         alt_exc, alt_inh) = _peak_to_peak_phase(
            exc_rate, inh_rate, peak_threshold, min_peak_dist)

        # Wrap into [0, 180] for symmetric reporting
        if mean_phase > 180:
            mean_phase = 360 - mean_phase

        peak_height_exc, peak_height_inh = _mean_peak_heights(
            exc_rate_raw, inh_rate_raw, peak_threshold, min_peak_dist)

        print(f'{pop_label} mean peak rate (E, I): '
              f'{peak_height_exc}, {peak_height_inh}')
        print(f'{pop_label} phase: {round(mean_phase, 2)} deg, '
              f'frequency: {round(mean_freq, 2)} Hz')
        print(f'{pop_label} phase CV: {round(phase_cv, 2)}%')

        return {
            'mean_phase_deg': round(mean_phase, 2),
            'phase_variance': round(phase_variance, 2),
            'phase_cv': round(phase_cv, 2),
            'mean_frequency_hz': round(mean_freq, 2),
            'mean_peak_height_exc': peak_height_exc,
            'mean_peak_height_inh': peak_height_inh,
            'alternating_exc_peaks': alt_exc,
            'alternating_inh_peaks': alt_inh,
        }

    except Exception as e:
        print(f'compute_phase_metrics failed for {pop_label}: {e}')
        return None


def _mean_peak_heights(sig1, sig2, min_peak_height, min_dist):
    """
    Mean peak height of two population rate signals.

    Peaks below `min_peak_height` (as a fraction of each signal's
    maximum) are ignored; peaks closer than `min_dist` samples are
    suppressed by `scipy.signal.find_peaks`.
    """
    threshold1 = np.max(sig1) * min_peak_height
    threshold2 = np.max(sig2) * min_peak_height

    _, props1 = find_peaks(
        sig1, height=threshold1, distance=min_dist, prominence=0.1)
    _, props2 = find_peaks(
        sig2, height=threshold2, distance=min_dist, prominence=0.1)

    return (round(float(np.mean(props1['peak_heights'])), 4),
            round(float(np.mean(props2['peak_heights'])), 4))


def _peak_to_peak_phase(sig1, sig2, min_peak_height, min_dist):
    """
    Estimate phase offset between two signals from alternating peaks.

    Detects peaks in both signals, then walks the two peak lists
    keeping only strictly alternating peaks (sig1, sig2, sig1, ...),
    discarding any that fall out of order. Phase offset is computed
    as the time difference between consecutive cross-signal peaks
    normalised by the average inter-peak period.

    Returns
    -------
    avg_phase_deg : float
        Mean phase offset in [0, 360) degrees.
    phase_variance : float
    phase_cv : float
        Coefficient of variation (%) of phase across pairs.
    avg_freq : float
        Mean oscillation frequency (Hz).
    alt1, alt2 : list of int
        Sample indices of the alternating peaks in each signal.
    """
    peaks1, _ = find_peaks(
        sig1, height=min_peak_height, distance=min_dist, prominence=0.1)
    peaks2, _ = find_peaks(
        sig2, height=min_peak_height, distance=min_dist, prominence=0.1)

    alt1, alt2 = _alternating_peaks(peaks1, peaks2)

    # Time differences and per-population periods
    time_diff = np.subtract(alt2, alt1)
    period1 = np.mean(np.diff(alt1)) * TIME_RESOLUTION
    period2 = np.mean(np.diff(alt2)) * TIME_RESOLUTION
    avg_period = (period1 + period2) / 2
    avg_freq = 1000 / avg_period

    # Phase as a fraction of the cycle, then degrees
    phase = (avg_period - (time_diff * TIME_RESOLUTION)) / avg_period
    phase_deg = phase * 360
    phase_variance = float(np.var(phase_deg))
    mean_phase_deg = float(np.mean(phase_deg))
    phase_cv = (np.std(phase_deg) / mean_phase_deg) * 100

    # Wrap into [0, 360)
    avg_phase_deg = (mean_phase_deg + 360) % 360

    return (round(avg_phase_deg, 2),
            round(phase_variance, 2),
            round(phase_cv, 2),
            round(avg_freq, 2),
            alt1, alt2)


def _alternating_peaks(peaks1, peaks2):
    """
    Walk two peak lists, returning only strictly alternating peaks.

    Pattern enforced: peaks1[a] < peaks2[b] < peaks1[c] < peaks2[d] ...
    Peaks that violate the pattern are skipped.

    Returns
    -------
    alt1, alt2 : list of int
        Aligned alternating peaks (truncated to common length).
    """
    alt1 = []
    alt2 = []

    i, j = 0, 0
    last_pop = None
    while i < len(peaks1) and j < len(peaks2):
        if last_pop is None or last_pop == 2:
            if peaks1[i] < peaks2[j]:
                alt1.append(peaks1[i])
                last_pop = 1
                i += 1
            else:
                j += 1
        elif last_pop == 1 and (peaks2[j] < peaks1[i]):
            alt2.append(peaks2[j])
            last_pop = 2
            j += 1
        else:
            i += 1

    n = min(len(alt1), len(alt2))
    return alt1[:n], alt2[:n]


# ======================================================================
# Cross-population helpers
# ======================================================================

def find_zero_overlap(arr1, arr2):
    """
    Total time during which both populations are silent simultaneously.

    Iterates through paired samples and accumulates the duration of
    intervals where both `arr1` and `arr2` are zero. Used as a proxy
    for inter-population suppression (e.g. PING-style alternation).

    Parameters
    ----------
    arr1, arr2 : 1-D array
        Population rate signals (must have equal length).

    Returns
    -------
    float
        Total duration (ms) of co-silent intervals.
    """
    if len(arr1) != len(arr2):
        raise ValueError('Arrays must have the same length')

    zero_durations = []
    duration = 0

    for i in range(len(arr1)):
        if arr1[i] == 0 and arr2[i] == 0:
            duration += 1
        elif duration > 0:
            zero_durations.append(duration)
            duration = 0

    return sum(zero_durations) * TIME_RESOLUTION
