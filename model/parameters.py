#!/usr/bin/env python
"""
Network parameters for the multilayer PFC microcircuit model.

This module defines the `neural_network` class, which loads simulation
parameters from `configuration_run_nest.yaml` and computes derived
quantities such as population sizes, per-pathway synaptic weights, and
spatial connectivity constants.

The class is instantiated once at simulation startup; downstream modules
(network construction, simulation control, analysis) import the resulting
parameter set rather than re-loading the YAML.

Population structure
--------------------
- 80% pyramidal neurons (PN), 20% inhibitory interneurons (4:1 E/I ratio).
- PN are split equally between L4 and L5.
- Inhibitory pool: 40% PV (L5), 30% SST (L4); remaining 30% (VIP/other)
  is unmodelled.

PV-PV connectivity is implemented via a two-stage model fitted to
Galarreta & Hestrin (2002) Table 1 paired-recording statistics.
See `network.connect_pv_pv` for full implementation.

Spatial geometry
----------------
1 NEST spatial unit = 200 microns. Layers are positioned along the
z-axis (L4 above z=0, L5 below) within a circular column of radius
0.5 units (~100 micron radius minicolumn).

References
----------
Galarreta & Hestrin (2002), PNAS 99: 12438-12443
Markram et al. (1997), J Physiol 500: 409-440
Pfeffer et al. (2013), Nat Neurosci 16: 1068-1076
Brunel (2000), J Comput Neurosci 8: 183-208
"""

import pathlib
import sys
import datetime

import numpy as np
import yaml


# Load YAML configuration once at module import time.
with open('configuration_run_nest.yaml') as _config_file:
    args = yaml.load(_config_file, Loader=yaml.FullLoader)
print('\nLoading parameters from configuration file:\n')


class neural_network:
    """
    Container for all network and simulation parameters.

    Parameters are loaded from `configuration_run_nest.yaml` and
    augmented with hard-coded biological constants. All attributes
    are set in `__init__`; downstream modules access them as
    `params = neural_network(); params.pv_pn_weight` etc.
    """

    def __init__(self):
        self.args = args
        self.optimizing = args['optimizing']
        self.gap_junctions_enabled = args['gap_junctions_enabled']

        # ------------------------------------------------------------------
        # Population sizes
        # ------------------------------------------------------------------
        self.pfc_pop_neurons = 5000

        # 4:1 E/I ratio (Rudy et al. 2011)
        self.ratio_exc_inh = 4

        # Top-level E/I split: 80% PN, 20% inhibitory
        self.pct_excitatory = 0.80
        self.pct_inhibitory = 0.20

        self.exc_neurons_count = int(np.round(
            self.pfc_pop_neurons * self.pct_excitatory))
        self.inh_neurons_count = self.pfc_pop_neurons - self.exc_neurons_count

        # PN split: equal halves between L4 and L5
        self.pct_pn_l4 = 0.50
        self.pct_pn_l5 = 0.50

        self.pn_l4_count = int(np.round(
            self.exc_neurons_count * self.pct_pn_l4))
        self.pn_exc_l5_count = self.exc_neurons_count - self.pn_l4_count

        # Legacy alias retained for backwards compatibility with network.py
        self.exc_bursting_count = self.pn_l4_count

        # Inhibitory split: 40% PV, 30% SST, 30% VIP/other (unmodelled)
        self.pct_pv = 0.40
        self.pct_sst = 0.30

        self.total_pv = int(np.round(self.inh_neurons_count * self.pct_pv))
        self.total_sst = int(np.round(self.inh_neurons_count * self.pct_sst))

        # SST: all SST neurons are placed in L4 in this model
        self.sst_inh_l4_count = self.total_sst

        # Legacy alias: total non-PN, non-PV-only-pair tonic interneurons
        self.inh_tonic_count = self.sst_inh_l4_count + self.total_pv

        # ------------------------------------------------------------------
        # PV-PV pair-type proportions (Galarreta & Hestrin 2002, Table 1)
        # ------------------------------------------------------------------
        # 23 adult PV-FS pairs in mouse somatosensory cortex (layers IV-V).
        # Six categories preserve chemical directionality (unidirectional
        # vs reciprocal) and gap-chemical co-occurrence.
        #
        #   43% gap + reciprocal chemical      (10/23)
        #   13% gap + unidirectional chemical  ( 3/23)
        #    4% gap only                       ( 1/23)
        #    0% reciprocal chemical only       ( 0/23)
        #   22% unidirectional chemical only   ( 5/23)
        #   17% neither                        ( 4/23)
        #
        # Reciprocal chemical without gap was never observed (0/23),
        # consistent with biological co-regulation of these connection
        # types.
        self.pv_pv_p_gap_recip_chem = 0.43
        self.pv_pv_p_gap_uni_chem = 0.13
        self.pv_pv_p_gap_only = 0.04
        self.pv_pv_p_recip_chem_only = 0.00
        self.pv_pv_p_uni_chem_only = 0.22
        # self.pv_pv_p_neither = 0.17 (implicit remainder)

        # Summary totals
        self.total_pn = self.pn_l4_count + self.pn_exc_l5_count
        self.total_modelled = (
            self.total_pn + self.total_sst + self.total_pv)
        self.total_unmodelled = (
            self.inh_neurons_count - self.total_sst - self.total_pv)

        print(
            f'\nPopulation sizes:'
            f'\n  PN L4:      {self.pn_l4_count}'
            f'\n  PN L5:      {self.pn_exc_l5_count}'
            f'\n  SST L4:     {self.sst_inh_l4_count}'
            f'\n  PV L5:      {self.total_pv}  '
            f'(6-cat pair wiring: 43/13/4/0/22/17%)'
            f'\n  VIP/other (not modelled): {self.total_unmodelled}'
            f'\n  Total modelled: {self.total_modelled}'
        )

        # ------------------------------------------------------------------
        # Per-pathway synaptic weights, delays, and spatial decay constants
        # ------------------------------------------------------------------
        # weight : base synaptic strength (nS). Negative = inhibitory.
        # delay  : axonal conduction delay (ms).
        # decay  : exponential decay length constant (NEST units,
        #          1 unit = 200 microns). Only used when spatial_weight
        #          is enabled. w(d) = weight * exp(-d / decay).

        # PV -> PN  (perisomatic basket-cell inhibition, ~80 micron axon)
        self.pv_pn_weight = -0.8
        self.pv_pn_delay = 0.5
        self.pv_pn_decay = 0.40

        # PN -> PV  (fast AMPA onto PV dendrites)
        self.pn_pv_weight = 1.0
        self.pn_pv_delay = 1.0
        self.pn_pv_decay = 0.80

        # PV -> PV  (chemical recurrence, sharper local decay ~60 microns)
        self.pv_pv_weight = -1.0
        self.pv_pv_delay = 1.0
        self.pv_pv_decay = 0.30

        # PN -> PN  (recurrent excitation, moderate falloff ~100 microns)
        self.pn_pn_weight = 0.3
        self.pn_pn_delay = 2.0
        self.pn_pn_decay = 0.80

        # PN -> SST  (excitatory drive onto SST interneurons)
        # Kawaguchi & Kubota 1997
        self.pn_sst_weight = 0.5
        self.pn_sst_delay = 2.0
        self.pn_sst_decay = 0.40

        # SST -> PN  (dendritic inhibition from SST)
        # Silberberg & Markram 2007: 0.5-1.5 nS range
        self.sst_pn_weight = -0.7
        self.sst_pn_delay = 2.0
        self.sst_pn_decay = 0.70

        # NOTE: SST -> PV cross-inhibition (Pfeffer et al. 2013) is
        # currently not modelled. To enable, uncomment and add the
        # corresponding nest.Connect call in network.py.
        # self.sst_pv_weight = -0.4
        # self.sst_pv_delay = 2.0
        # self.sst_pv_decay = 0.40

        # ------------------------------------------------------------------
        # Simulation parameters
        # ------------------------------------------------------------------
        # Random seed: provided as command-line arg or via YAML
        if len(sys.argv) > 1:
            self.rng_seed = int(sys.argv[1])
        else:
            self.rng_seed = args['seed']

        self.time_resolution = args['delta_clock']  # ms
        self.sim_time = args['t_steps']             # ms

        # ------------------------------------------------------------------
        # Neuronal parameters (AdEx)
        # ------------------------------------------------------------------
        # Threshold and reset
        self.V_th_mean_tonic = -50.0     # mV
        self.V_th_std_tonic = 1.0
        self.V_th_mean_bursting = -51.0
        self.V_th_std_bursting = 1.0
        self.V_m_mean = -60.0
        self.V_m_std = 10.0

        # Membrane capacitances
        self.C_m_tonic_mean = 200.0       # pF -- SST / regular-spiking PNs
        self.C_m_tonic_fast_mean = 100.0  # pF -- PV fast-spiking
        self.C_m_tonic_std = 40.0
        self.C_m_bursting_mean = 250.0    # pF -- L4 bursting PNs
        self.C_m_bursting_std = 80.0

        # Refractory periods
        self.t_ref_mean = 9.0             # ms -- SST / tonic
        self.t_ref_std = 0.2
        self.t_ref_bursting_mean = 3.0    # ms -- L4/L5 PNs
        self.t_ref_bursting_std = 0.2
        self.t_ref_tonic_fast_mean = 1.5  # ms -- PV fast-spiking
        self.t_ref_tonic_fast_std = 0.2

        # Synaptic time constants (beta function)
        self.tau_syn_e_rise = 0.2
        self.tau_syn_e_decay = 1.0
        self.tau_syn_i_rise = 0.5
        self.tau_syn_i_decay = 3.0

        self.synaptic_delay = 2.0         # ms

        # ------------------------------------------------------------------
        # Poisson background input
        # ------------------------------------------------------------------
        # One independent Poisson generator per neuron (avoids shared-noise
        # synchrony artefact). Rates represent aggregate thalamocortical +
        # corticocortical bombardment; weight is excitatory in nS.
        # Brunel 2000, Wang 2010: high-conductance state requires ~thousands
        # of background events/s to keep neurons near threshold.
        self.poisson_rate_pn_l4 = 1500.0  # Hz -- stronger thalamic input to L4
        self.poisson_rate_pn_l5 = 1300.0  # Hz -- L5 PN tonic drive
        self.poisson_rate_pv = 10.0       # Hz -- PV driven mainly by PN volleys
        self.poisson_rate_sst = 200.0     # Hz -- moderate input to SST
        self.poisson_weight = 0.1         # nS

        # Effective inhibitory drive onto L5 PN representing missing
        # SST/VIP populations (currently disabled; only PV is modelled
        # as inhibitory source for L5 PN).
        self.poisson_rate_pn_l5_inh = 0.0
        self.poisson_weight_inh = -0.1

        # ------------------------------------------------------------------
        # Background noise
        # ------------------------------------------------------------------
        self.noise_mean = 10.0
        self.noise_std = 0.5

        # ------------------------------------------------------------------
        # Analysis / plotting parameters (passed through from YAML)
        # ------------------------------------------------------------------
        self.convstd_rate = args['convstd_rate']
        self.chop_edges_amount = args['chop_edges_amount']
        self.remove_mean = args['remove_mean']
        self.high_pass_filtered = args['high_pass_filtered']
        self.downsampling_convolved = args['downsampling_convolved']
        self.remove_silent = args['remove_silent']
        self.calculate_balance = args['calculate_balance']
        self.raster_plot = args['raster_plot']
        self.rate_coded_plot = args['rate_coded_plot']
        self.isf_output = args['isf_plot']
        self.membrane_potential_plot = args['membrane_potential_plot']
        self.plot_neuron_positions = args['plot_neuron_positions']
        self.time_window = args['smoothing_window']
        self.time_window_indiv_neurons = args['smoothing_window_indiv_neurons']

        # Spike detector parameters
        self.sd_params = {
            'withtime': True,
            'withgid': True,
            'to_file': False,
            'flush_after_simulate': False,
            'flush_records': True,
        }

        # ------------------------------------------------------------------
        # Spatial geometry (3D positioning)
        # ------------------------------------------------------------------
        # 1 NEST spatial unit = 200 microns
        # Horizontal: circular minicolumn
        # Vertical:   L4 above z=0, L5 below
        self.spatial_sparsity = args['spatial_sparsity']
        self.spatial_radius = 0.5         # ~100 micron radius minicolumn

        self.l4_z_min = 0.0
        self.l4_z_max = 0.5               # 100 micron thick
        self.l5_z_min = -1.0              # 200 micron thick
        self.l5_z_max = 0.0

        # ------------------------------------------------------------------
        # Per-pathway spatial connectivity parameters
        # ------------------------------------------------------------------
        # Used when spatial_sparsity == 1.
        # beta:  exponential decay length constant (NEST units)
        #        p(d) = p_max * exp(-d / beta)
        # p_max: peak connection probability at d = 0
        # p_flat: flat probability used when spatial_sparsity == 0

        # Within L4: PN <-> SST
        self.pn_sst_l4_beta = 0.40
        self.pn_sst_l4_pmax = 0.25
        self.pn_sst_l4_pflat = 0.03
        self.sst_pn_l4_beta = 0.40
        self.sst_pn_l4_pmax = 0.30
        self.sst_pn_l4_pflat = 0.03

        # Within-layer PN recurrent
        # Sparse recurrence; Markram et al. 1997: axon collaterals 500+ microns
        self.pn_l4_self_beta = 0.50
        self.pn_l4_self_pmax = 0.10
        self.pn_l4_self_pflat = 0.50
        self.pn_l5_self_beta = 0.60
        self.pn_l5_self_pmax = 0.10
        self.pn_l5_self_pflat = 0.50

        # Cross-layer: L4 -> L5 feedforward
        # The z-offset further reduces effective probability under
        # exponential distance gating.
        self.pn_l4_pn_l5_ff_beta = 0.4
        self.pn_l4_pn_l5_ff_pmax = 0.05
        self.pn_l4_pn_l5_ff_pflat = 0.05

        # Cross-layer: PN bidirectional recurrent (currently disabled)
        self.pn_l4_pn_l5_rec_beta = 0.4
        self.pn_l4_pn_l5_rec_pmax = 0.0
        self.pn_l4_pn_l5_rec_pflat = 0.50
        self.pn_l5_pn_l4_rec_beta = 0.4
        self.pn_l5_pn_l4_rec_pmax = 0.0
        self.pn_l5_pn_l4_rec_pflat = 0.50

        # ------------------------------------------------------------------
        # PV-PV two-stage connectivity parameters
        # ------------------------------------------------------------------
        # Stage 1: distance-dependent gate (does the pair connect at all?)
        #   p_connect(d) = p_connect_at_d0 * exp(-d / pv_pv_conn_beta)
        #   p_connect_at_d0 derived from Galarreta Table 1: 19/23 = 0.826
        #   When spatial_sparsity == 0: p_connect = p_connect_at_d0 (flat)
        #
        # Stage 2: conditional category assignment (what type of connection?)
        #   Sampled from Table 1 renormalised over connected categories.
        #   Category proportions are distance-INDEPENDENT (data-driven).
        #
        # This ensures spatial decay only affects total connection count,
        # NOT the relative proportions of connection types.
        self.pv_pv_conn_beta = 0.40

        # Gap junction conductance (per junction, nS)
        self.pv_gap_weight = 0.2

        if args['spatial_weight'] == 1:
            print('Spatial weight mode ON: weights decay exponentially '
                  'with distance')
        else:
            print('Spatial weight mode OFF: flat base weights')

        if self.spatial_sparsity == 0:
            print('Sparsity is assigned with a distribution (flat mode)')
        else:
            print('Spatial parameters define sparsity (per-pathway 3D mode)')

        # Multimeter parameters
        self.mm_params = {'interval': 0.1, 'record_from': ['V_m']}

        # ------------------------------------------------------------------
        # Output directory setup
        # ------------------------------------------------------------------
        # When `save_results` is enabled and we are not in optimisation
        # mode, create a timestamped output folder for this simulation
        # run and dump the active YAML configuration alongside it for
        # reproducibility. The paths are exposed as `self.path` and
        # `self.pathFigures` for downstream use by run.py.
        if args['save_results'] and not args['optimizing']:
            if len(sys.argv) > 1:
                run_id = (
                    str(int(sys.argv[1])) + '_'
                    + datetime.datetime.now().strftime(
                        '%Y-%m-%d-%H-%M-%S'))
            else:
                run_id = datetime.datetime.now().strftime(
                    '%Y-%m-%d-%H-%M-%S')

            self.path = 'saved_simulations/' + run_id
            self.pathFigures = self.path + '/Figures'
            pathlib.Path(self.path).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.pathFigures).mkdir(
                parents=True, exist_ok=True)

            with open(self.path + '/args_' + run_id + '.yaml', 'w') as f:
                yaml.dump(args, f)
        else:
            self.path = None
            self.pathFigures = None
