#!/usr/bin/env python
"""
Population-level spike processing utilities.

This module provides functions for converting NEST spike-recorder
output into population firing rate signals, computing per-neuron
firing statistics, and applying smoothing and bandpass filtering
that is consistent across analysis pipelines.

Function groups
---------------
- Spike data extraction: `read_spike_data`, `read_recent_spike_data`,
  `read_membrane_potential`, `save_spike_data`.
- Per-neuron summaries: `count_indiv_spikes`, `single_neuron_spikes`,
  `single_neuron_spikes_binary`.
- Interspike-frequency analysis: `calculate_interspike_frequency`,
  `calculate_avg_interspike_frequencies`.
- Population rate construction: `rate_code_spikes`,
  `convolve_spiking_activity`.
- Smoothing primitives: `sliding_time_window`,
  `sliding_time_window_matrix`, `padded_sliding_time_window`, `smooth`.
- Network manipulation: `update_neuronal_characteristic`,
  `inject_current`.

All functions read the simulation parameter object via the module-
level `nn` handle, which is set at import time from
`set_network_params.neural_network`.
"""

import copy

import nest
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt, decimate, convolve, windows

from model.parameters import neural_network


nn = neural_network()


# ======================================================================
# Spike data extraction
# ======================================================================

def read_spike_data(spike_detector):
    """
    Pull all senders and spike times from a NEST spike recorder.

    Parameters
    ----------
    spike_detector : nest.NodeCollection
        A NEST `spike_recorder` node (or collection of one).

    Returns
    -------
    senders : list
        One element per call: an array of sender GIDs from the events.
    spiketimes : list
        One element per call: an array of spike times (ms).
    """
    senders = [spike_detector.get('events', 'senders')]
    spiketimes = [spike_detector.get('events', 'times')]
    return senders, spiketimes


def read_recent_spike_data(spike_detector, window_ms=1.0):
    """
    Count spikes that occurred within the last `window_ms` of biological time.

    Useful for online stability checks during a long simulation:
    a population that has stopped firing will return zero here even
    if the recorder has accumulated many earlier events.
    """
    spiketimes = [spike_detector.get('events', 'times')]
    current_time = nest.biological_time
    total_spikes = 0
    for arr in spiketimes:
        total_spikes += int(np.sum(
            (arr >= current_time - window_ms) & (arr <= current_time)))
    return total_spikes


def read_membrane_potential(multimeter, pop_size, neuron_num):
    """
    Extract the V_m trace for a single neuron from a multimeter recording.

    Multimeter events interleave samples from all connected neurons, so
    we slice with a stride of `pop_size` to pull out the trace for the
    specified `neuron_num`. Samples are returned in chronological order.

    Parameters
    ----------
    multimeter : nest.NodeCollection
        The multimeter node.
    pop_size : int
        Number of neurons connected to this multimeter (stride).
    neuron_num : int
        Index of the neuron of interest within the connected population.

    Returns
    -------
    vm, t_vm : 1-D arrays
        Membrane potential samples and corresponding times (ms).
    """
    mm = nest.GetStatus(multimeter, keys='events')[0]
    vm = mm['V_m'][neuron_num::pop_size]
    t_vm = mm['times'][neuron_num::pop_size]

    sorted_indices = np.argsort(t_vm)
    return vm[sorted_indices], t_vm[sorted_indices]


def save_spike_data(num_neurons, population, neuron_num_offset):
    """
    Flatten per-neuron spike trains into a list of (neuron_id, time) pairs.

    Parameters
    ----------
    num_neurons : int
    population : sequence
        Output from `read_spike_data` (i.e. `population[0][i]` is the
        spike-time array for neuron `i`).
    neuron_num_offset : int
        Added to each neuron index so that combined dumps from
        multiple populations remain distinguishable.

    Returns
    -------
    list of tuples
        Each tuple is (neuron_id, spike_time_ms).
    """
    all_spikes = []
    for i in range(num_neurons):
        spike_data = population[0][i]
        neuron_id = i + neuron_num_offset
        all_spikes.extend([(neuron_id, t) for t in spike_data])
    return all_spikes


# ======================================================================
# Per-neuron summaries
# ======================================================================

def count_indiv_spikes(total_neurons, neuron_id_data, calc_freq):
    """
    Categorise individual neurons by their total spike count.

    A "sparse-firing" neuron is one whose count falls between 1 and the
    threshold derived from the population's calculated frequency
    (`2 * calc_freq` spikes per second of simulation). A silent neuron
    has zero spikes. Neurons firing above the sparse threshold are
    treated as densely active.

    Parameters
    ----------
    total_neurons : int
    neuron_id_data : sequence
        Output from `read_spike_data`.
    calc_freq : float
        Population oscillation frequency in Hz.

    Returns
    -------
    spike_count_array : list of int
    neuron_to_sample : int
        Index of a sparse-firing neuron suitable for plotting; falls
        back to neuron 0 if no sparse-firing neurons are found.
    n_sparse : int
    n_silent : int
    """
    import math
    total_spikes_per_second = (
        6 if math.isnan(calc_freq) else int(calc_freq * 2))
    spike_count_array = [
        len(neuron_id_data[0][i]) for i in range(total_neurons)]
    sparse_count_max = total_spikes_per_second * (nn.sim_time / 1000)
    sparse_firing_count = [
        i for i, count in enumerate(spike_count_array)
        if 1 <= count <= sparse_count_max]
    silent_neuron_count = [
        i for i, count in enumerate(spike_count_array) if count == 0]
    neuron_to_sample = (
        sparse_firing_count[1] if len(sparse_firing_count) > 1 else 0)
    return (spike_count_array,
            neuron_to_sample,
            len(sparse_firing_count),
            len(silent_neuron_count))


def single_neuron_spikes(neuron_number, population):
    """
    Build a sparse spike-time array for a single neuron.

    The output array has one entry per simulation timestep
    (`sim_time / time_resolution`). Indices corresponding to spike
    times are set to the spike time itself; all other indices are 0.

    Use `single_neuron_spikes_binary` for a 0/1 representation.
    """
    n_steps = int(nn.sim_time / nn.time_resolution)
    spike_time = [0] * n_steps
    spike_data = population[0][neuron_number]
    for t in spike_data:
        idx = int(t * (1 / nn.time_resolution)) - 1
        spike_time[idx] = t
    return spike_time


def single_neuron_spikes_binary(neuron_number, population):
    """
    Build a 0/1 spike train for a single neuron at simulation resolution.

    Returns
    -------
    list of int
        Length = `sim_time / time_resolution`, with 1 at every
        spike-time index and 0 elsewhere.
    """
    n_steps = int(nn.sim_time / nn.time_resolution)
    spike_time = [0] * n_steps
    spike_data = population[0][neuron_number]
    for t in spike_data:
        idx = int(t * (1 / nn.time_resolution)) - 1
        spike_time[idx] = 1
    return spike_time


# ======================================================================
# Interspike-frequency analysis
# ======================================================================

def calculate_interspike_frequency(neuron_count, output_spiketimes):
    """
    Per-neuron instantaneous frequency from interspike intervals.

    For each neuron with at least two spikes, computes the
    instantaneous frequency f = 1000 / ISI for every consecutive
    pair of spikes. Neurons with fewer than two spikes return
    `[NaN]` placeholders so that output indexing remains aligned
    with the input.

    Parameters
    ----------
    neuron_count : int
    output_spiketimes : sequence
        Output of `read_spike_data`.

    Returns
    -------
    frequencies : list of arrays
        Per-neuron instantaneous frequencies (Hz).
    times : list of arrays
        Times at which each frequency was evaluated (ms).
    """
    frequencies = []
    times = []
    for i in range(neuron_count):
        t_spikes = output_spiketimes[0][i]
        if len(t_spikes) > 1:
            spike_times = np.sort(t_spikes)
            isi = np.diff(spike_times)
            valid = ~np.isnan(isi)
            valid_isi = isi[valid]
            valid_times = spike_times[1:][valid]
            if len(valid_isi) > 0:
                frequencies.append(1000.0 / valid_isi)
                times.append(valid_times)
            else:
                frequencies.append(np.array([np.nan]))
                times.append(np.array([np.nan]))
        else:
            frequencies.append(np.array([np.nan]))
            times.append(np.array([np.nan]))
    return frequencies, times


def calculate_avg_interspike_frequencies(output_spiketimes):
    """
    Population-averaged instantaneous frequency, binned over time.

    Pools instantaneous frequencies from all neurons, bins them by
    time of occurrence (bin width = `nn.time_window * nn.time_resolution`
    ms), and returns a smoothed time series of mean frequency per bin.
    Empty bins are filled by averaging the nearest non-empty bins on
    either side.

    Returns
    -------
    smoothed_freqs : 1-D array
        Gaussian-smoothed mean frequency per bin (Hz).
    bin_starts : 1-D array
        Left edge of each bin (ms).
    """
    total_time = nn.sim_time
    bin_width = nn.time_window * nn.time_resolution
    bin_edges = np.arange(0, total_time + bin_width, bin_width)
    num_bins = len(bin_edges) - 1

    # Gather all instantaneous frequencies and their evaluation times
    all_times = []
    all_freqs = []
    for neuron_spikes in output_spiketimes[0]:
        if len(neuron_spikes) > 1:
            sorted_spikes = np.sort(neuron_spikes)
            isi = np.diff(sorted_spikes)
            freqs = 1000.0 / isi
            times = sorted_spikes[1:]
            all_freqs.extend(freqs)
            all_times.extend(times)

    all_freqs = np.array(all_freqs)
    all_times = np.array(all_times)

    # Bin and accumulate
    bin_sums = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    bin_indices = np.digitize(all_times, bin_edges) - 1
    for i, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < num_bins:
            bin_sums[bin_idx] += all_freqs[i]
            bin_counts[bin_idx] += 1

    # Compute averages, ignoring divide-by-zero warnings for empty bins
    with np.errstate(invalid='ignore'):
        avg_freqs = np.divide(bin_sums, bin_counts, where=bin_counts != 0)

    # Fill empty bins by averaging nearest non-empty neighbours
    for i in range(num_bins):
        if bin_counts[i] == 0:
            prev_val = next_val = None
            for j in range(i - 1, -1, -1):
                if bin_counts[j] != 0:
                    prev_val = avg_freqs[j]
                    break
            for j in range(i + 1, num_bins):
                if bin_counts[j] != 0:
                    next_val = avg_freqs[j]
                    break
            if prev_val is not None and next_val is not None:
                avg_freqs[i] = (prev_val + next_val) / 2
            elif prev_val is not None:
                avg_freqs[i] = prev_val
            elif next_val is not None:
                avg_freqs[i] = next_val
            else:
                avg_freqs[i] = 0

    smoothed_freqs = gaussian_filter(avg_freqs, 2)
    return smoothed_freqs, bin_edges[:-1]


# ======================================================================
# Population rate construction
# ======================================================================

def rate_code_spikes(neuron_count, output_spiketimes):
    """
    Convert spike trains into a smoothed population firing rate.

    Pipeline:
        1. Bin spikes for each neuron at simulation resolution.
        2. Sum across neurons to a population spike count per bin.
        3. Apply a sliding-window sum to smooth at `nn.time_window`.
        4. Apply a Gaussian filter (sigma `nn.convstd_rate`) to
           remove residual high-frequency noise.
        5. Optionally chop edge artefacts (`nn.chop_edges_amount`).

    Parameters
    ----------
    neuron_count : int
    output_spiketimes : sequence
        Output of `read_spike_data`.

    Returns
    -------
    1-D array
        Smoothed population firing rate (a.u., proportional to
        spikes per `time_window`).
    """
    bins = np.arange(
        0, nn.sim_time + nn.time_resolution, nn.time_resolution)

    spike_bins_current = None
    for i in range(neuron_count):
        t_spikes = output_spiketimes[0][i]
        spikes_per_bin, _ = np.histogram(t_spikes, bins)
        if spike_bins_current is None:
            spike_bins_current = spikes_per_bin
        else:
            spike_bins_current += spikes_per_bin

    # Apply sliding-window sum then a Gaussian filter
    spike_bins_current = sliding_time_window(spike_bins_current, nn.time_window)
    smoothed_spike_bins = gaussian_filter(spike_bins_current, nn.convstd_rate)

    if nn.chop_edges_amount > 0.0:
        chop = int(nn.chop_edges_amount)
        smoothed_spike_bins = smoothed_spike_bins[chop:-chop]

    return smoothed_spike_bins


def convolve_spiking_activity(population_size, population):
    """
    Build a smoothed and (optionally) bandpass-filtered population rate signal.

    Equivalent to `rate_code_spikes` for the population sum, but
    additionally:
        - operates on a per-neuron binary spike matrix
        - supports baseline mean removal (`nn.remove_mean`)
        - supports high-pass filtering (`nn.high_pass_filtered`)
        - supports downsampling (`nn.downsampling_convolved`)

    The high-pass filter uses the same parameters as Linden et al. 2022
    (3rd-order Butterworth at 0.1 normalised cutoff, fs=1000).

    Parameters
    ----------
    population_size : int
    population : sequence
        Output of `read_spike_data`.

    Returns
    -------
    smoothed_spikes : 1-D array
        Population-mean rate signal after all enabled processing steps.
    time_vector : 1-D array
        Corresponding time axis (ms).
    """
    binary_spikes = np.vstack(
        [single_neuron_spikes_binary(i, population)
         for i in range(population_size)])
    binned_spikes = sliding_time_window_matrix(binary_spikes, nn.time_window)
    smoothed_spikes = smooth(binned_spikes, nn.convstd_rate)

    time_vector = np.arange(binned_spikes.shape[1]) * nn.time_resolution

    if nn.chop_edges_amount > 0.0:
        chop = int(nn.chop_edges_amount)
        smoothed_spikes = smoothed_spikes[:, chop:-chop]
        time_vector = time_vector[chop:-chop]

    if nn.remove_mean:
        smoothed_spikes = (
            smoothed_spikes.T - np.mean(smoothed_spikes, axis=1)).T

    if nn.high_pass_filtered:
        # Linden et al. 2022: 3rd-order Butterworth, normalised cutoff 0.1
        b, a = butter(3, 0.1, 'highpass', fs=1000)
        smoothed_spikes = filtfilt(b, a, smoothed_spikes)

    if nn.downsampling_convolved:
        decimation_factor = int(1 / nn.time_resolution)
        smoothed_spikes = decimate(
            smoothed_spikes, decimation_factor, n=2,
            ftype='iir', zero_phase=True)
        time_vector = time_vector[::decimation_factor]

    # Truncate by the width of the time window (alignment with smoothing)
    smoothed_spikes = smoothed_spikes[:, :-nn.time_window + 1]
    time_vector = time_vector[:smoothed_spikes.shape[1]]

    return smoothed_spikes.mean(axis=0), time_vector


# ======================================================================
# Smoothing primitives
# ======================================================================

def sliding_time_window(signal, window_size):
    """Sum over a sliding window of `window_size` samples (1-D)."""
    win = np.lib.stride_tricks.sliding_window_view(signal, window_size)
    return np.sum(win, axis=1)


def sliding_time_window_matrix(signal, window_size):
    """Sum over a sliding window of `window_size` samples (2-D, per row)."""
    result = []
    for row in signal:
        win = np.lib.stride_tricks.sliding_window_view(row, window_size)
        result.append(np.sum(win, axis=1))
    return np.array(result)


def padded_sliding_time_window(signal, window_size):
    """
    Mean over a sliding window with edge-padding to preserve length.

    The input signal is padded by `window_size // 2` samples on each
    side using edge values, so the output has the same length as the
    input.
    """
    padded = np.pad(
        signal, (window_size // 2, window_size // 2), mode='edge')
    win = np.lib.stride_tricks.sliding_window_view(padded, window_size)
    return np.mean(win, axis=1)[:len(signal)]


def smooth(data, sd):
    """
    Gaussian smoothing along each row of a 2-D array.

    Uses scipy.signal.convolve in 'same' mode so the output has the
    same shape as the input. Window length is set to the largest odd
    number not exceeding the number of bins, ensuring symmetric
    convolution.

    Parameters
    ----------
    data : 2-D array
        Each row is smoothed independently.
    sd : float
        Standard deviation of the Gaussian kernel (in samples).

    Returns
    -------
    2-D array
        Smoothed data, same shape as input.
    """
    data = copy.copy(data)
    n_bins = data.shape[1]
    w = n_bins - 1 if n_bins % 2 == 0 else n_bins
    window = windows.gaussian(w, std=sd)
    for j in range(data.shape[0]):
        data[j, :] = convolve(
            data[j, :], window, mode='same', method='auto')
    return data


# ======================================================================
# Network manipulation
# ======================================================================

def update_neuronal_characteristic(charac_name, neuron_population, value):
    """
    Set a parameter (e.g. 'g_L', 'V_th') to `value` for every neuron in
    `neuron_population`. Returns the new value, read back from the first
    neuron as a sanity check.
    """
    for neuron in neuron_population:
        nest.SetStatus(neuron, {charac_name: value})
    return nest.GetStatus(neuron_population, keys=charac_name)[0]


def inject_current(neuron_population, current):
    """
    Set the constant input current `I_e` (pA) for every neuron in
    `neuron_population`. Returns the value read back from the first
    neuron.
    """
    for neuron in neuron_population:
        nest.SetStatus([neuron], {'I_e': current})
    return nest.GetStatus(neuron_population, keys='I_e')[0]


def normalize_rows(matrix):
    """Divide each row of a 2-D array by its maximum element."""
    max_values = np.max(matrix, axis=1, keepdims=True)
    return matrix / max_values
