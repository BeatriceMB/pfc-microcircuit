#!/usr/bin/env python
"""
Simulation entry point for the multilayer PFC model.

This script runs a single simulation of the PFC microcircuit:

    1. Initialise the NEST kernel and install the AdEx + gap-junction module.
    2. Build the network (neurons, recorders, synapses) via
       `create_multilayer_pfc_spatial.create_pfc_network`.
    3. Run the simulation in 10-step batches with a progress indicator.
    4. Read spike data and membrane potentials from all populations.
    5. Pickle the simulation results to disk for downstream plotting.

Usage
-----
    python run.py [seed]

If a positive integer is supplied as the first argument, it overrides
the seed defined in `configuration_run_nest.yaml`.

Output
------
A single pickle file is written to the timestamped output directory
(`saved_simulations/<id>/sim_data.pkl`) containing:

    - spike senders and spike times for every population
    - membrane potential traces for the sampled neuron in each population
    - PV-PV pair-type masks and per-neuron connectivity profiles
    - random seed and selected configuration values

The plotting script (`plot.py`) reads this file to reproduce all
figures without re-running the simulation.
"""

import time
import pickle
from pathlib import Path

import numpy as np
import nest

from model import simulation as ss
from model import parameters as netparams
from analysis import population as popfunc
from model import network as pfc


# Initialise NEST and load parameters.
ss.nest_start()
nn = netparams.neural_network()


def _read_single_vm(multimeter):
    """Read V_m from a multimeter connected to exactly one neuron."""
    mm = nest.GetStatus(multimeter, keys='events')[0]
    vm = np.array(mm['V_m'])
    t_vm = np.array(mm['times'])
    idx = np.argsort(t_vm)
    return vm[idx], t_vm[idx]


def main():
    # ------------------------------------------------------------------
    # Build the network
    # ------------------------------------------------------------------
    pfc1 = pfc.create_pfc_network()

    print(f'Seed#: {nn.rng_seed}')
    print(f'PN L4 count:   {nn.pn_l4_count}')
    print(f'PN L5 count:   {nn.pn_exc_l5_count}')
    print(f'SST L4 count:  {nn.sst_inh_l4_count}')
    print(f'PV L5 count:   {nn.total_pv}')

    # ------------------------------------------------------------------
    # Run the simulation in batches with a progress indicator
    # ------------------------------------------------------------------
    init_time = 50  # ms -- short warm-up before the timed loop
    nest.Simulate(init_time)

    num_steps = int(nn.sim_time / nn.time_resolution)
    n_batches = int(num_steps / 10) - init_time

    t_start = time.perf_counter()
    for _ in range(n_batches):
        nest.Simulate(nn.time_resolution * 10)
        print(f't = {nest.biological_time}', end='\r')
    t_stop = time.perf_counter()

    print(f'\nSimulation completed in {round(t_stop - t_start, 2)} seconds.')

    # ------------------------------------------------------------------
    # Read spike data from every population
    # ------------------------------------------------------------------
    senders_pn_l4, spiketimes_pn_l4 = popfunc.read_spike_data(
        pfc1.spike_detector_pn_exc_l4)
    senders_pn_l5, spiketimes_pn_l5 = popfunc.read_spike_data(
        pfc1.spike_detector_pn_exc_l5)
    senders_sst_l4, spiketimes_sst_l4 = popfunc.read_spike_data(
        pfc1.spike_detector_sst_inh_l4)
    senders_pv_l5, spiketimes_pv_l5 = popfunc.read_spike_data(
        pfc1.spike_detector_pv_inh_l5)

    # ------------------------------------------------------------------
    # Read membrane potential traces (one sample neuron per population)
    # ------------------------------------------------------------------
    vm_pn_l4, t_vm_pn_l4 = _read_single_vm(pfc1.mm_pn_exc_l4)
    vm_pn_l5, t_vm_pn_l5 = _read_single_vm(pfc1.mm_pn_exc_l5)
    vm_sst_l4, t_vm_sst_l4 = _read_single_vm(pfc1.mm_sst_inh_l4)
    vm_pv_l5, t_vm_pv_l5 = _read_single_vm(pfc1.mm_pv_inh_l5)

    # ------------------------------------------------------------------
    # Bundle results for plotting and save to disk
    # ------------------------------------------------------------------
    sim_data = {
        # Configuration
        'rng_seed': nn.rng_seed,
        'sim_time': nn.sim_time,
        'time_resolution': nn.time_resolution,
        'pn_l4_count': nn.pn_l4_count,
        'pn_l5_count': nn.pn_exc_l5_count,
        'sst_l4_count': nn.sst_inh_l4_count,
        'pv_l5_count': nn.total_pv,
        'mm_neuron_idx': pfc1.mm_neuron_idx,

        # Spike data
        'senders_pn_l4': senders_pn_l4,
        'spiketimes_pn_l4': spiketimes_pn_l4,
        'senders_pn_l5': senders_pn_l5,
        'spiketimes_pn_l5': spiketimes_pn_l5,
        'senders_sst_l4': senders_sst_l4,
        'spiketimes_sst_l4': spiketimes_sst_l4,
        'senders_pv_l5': senders_pv_l5,
        'spiketimes_pv_l5': spiketimes_pv_l5,

        # Membrane potentials
        'vm_pn_l4': vm_pn_l4, 't_vm_pn_l4': t_vm_pn_l4,
        'vm_pn_l5': vm_pn_l5, 't_vm_pn_l5': t_vm_pn_l5,
        'vm_sst_l4': vm_sst_l4, 't_vm_sst_l4': t_vm_sst_l4,
        'vm_pv_l5': vm_pv_l5, 't_vm_pv_l5': t_vm_pv_l5,

        # Spatial and connectivity
        'positions': {
            'pn_l4': pfc1.pn_exc_l4_positions,
            'pn_l5': pfc1.pn_exc_l5_positions,
            'sst_l4': pfc1.sst_inh_l4_positions,
            'pv_l5': pfc1.pv_inh_l5_positions,
        },
        'pv_pv_pair_types': pfc1.pv_pv_pair_types,
        'pv_profiles': pfc1.pv_profiles,
    }

    if nn.args['save_results'] and not nn.args['optimizing']:
        out_path = Path(nn.path) / 'sim_data.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(sim_data, f)
        print(f'\nSimulation data written to {out_path}')
        print(f'To plot: python plot.py {out_path}')
    else:
        print('\nsave_results disabled; sim_data not pickled.')

    return sim_data


if __name__ == '__main__':
    main()
