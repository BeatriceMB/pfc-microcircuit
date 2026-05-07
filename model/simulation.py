#!/usr/bin/env python
"""
NEST kernel initialisation for the multilayer PFC simulation.

This module exposes the `nest_start` class, which configures the NEST
kernel before the network is built. It performs the following steps:

    1. Reset the NEST kernel to a clean state.
    2. Install the `nestml_gap_aeif_beta_module` providing the AdEx
       neuron with beta-function synapses and gap junction support.
    3. Set verbosity, threading, time resolution, and the master RNG seed.

The class is instantiated once at the start of every simulation run,
before any neuron populations are created.
"""

import sys

import numpy as np
import nest

from model.parameters import neural_network


netparams = neural_network()


class nest_start:
    """
    Configure the NEST kernel for a single simulation run.

    Reads simulation parameters (time resolution, RNG seed, thread count)
    from the module-level `netparams` object and applies them to the
    active kernel. The NESTML module providing AdEx neurons with beta-
    function synapses and gap junctions is installed at construction
    time, so any subsequent `nest.Create` call can use the
    `aeif_cond_beta_nestml` model and `gap_junction` synapse model.
    """

    # Number of OpenMP threads used for parallel simulation. Adjust to
    # match the host machine if running on a different system.
    DEFAULT_THREADS = 6

    def __init__(self):
        np.set_printoptions(precision=1, threshold=sys.maxsize)

        nest.ResetKernel()
        nest.Install('nestml_gap_aeif_beta_module')
        nest.set_verbosity('M_ERROR')
        nest.SetKernelStatus({
            'local_num_threads': self.DEFAULT_THREADS,
            'resolution': netparams.time_resolution,
            'rng_seed': netparams.rng_seed,
        })
