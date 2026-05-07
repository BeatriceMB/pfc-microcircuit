#!/usr/bin/env python
"""
Construction of the multilayer PFC spatial network.

This module builds the spiking PFC microcircuit in NEST, comprising:
  - L4 pyramidal neurons (PN) and SST interneurons
  - L5 pyramidal neurons (PN) and PV interneurons

Three connectivity schemes are implemented:

1. Standard chemical synapses with per-pathway distance-dependent
   probability and (optionally) distance-dependent weight decay.
   Used for PN-PN recurrence, PN-SST, and SST-PN.

2. PV-PN connectivity with empirical reciprocal overrepresentation,
   sampled from paired-recording statistics rather than a single
   bulk Bernoulli probability. See `connect_pv_pn`.

3. PV-PV connectivity using a two-stage model fitted to Galarreta &
   Hestrin (2002) Table 1: a distance-dependent connectivity gate
   followed by distance-independent motif assignment. Preserves the
   joint distribution of gap-junction and chemical synapse motifs.
   See `connect_pv_pv`.

The main entry point is the `create_pfc_network` class, which builds
the network in a sequence of clearly delimited stages.

References
----------
Galarreta & Hestrin (2002), PNAS 99: 12438-12443
Otsuka & Kawaguchi (2009), J Neurosci 29: 10533-10540
Markram et al. (1997), J Physiol 500: 409-440
Brunel (2000), J Comput Neurosci 8: 183-208
"""

import numpy as np
import nest
from scipy.spatial.distance import squareform, pdist

from model.parameters import neural_network


# Single module-level instance retained for convenience. Downstream
# functions reference netparams.* directly. To use a different parameter
# set, replace this binding before importing dependent modules.
netparams = neural_network()


# ======================================================================
# Module-level helpers
# ======================================================================

def _make_layer_positions(n, z_min, z_max):
    """
    Generate 3D positions uniformly distributed within a circular disk
    in the (x, y) plane at a given vertical (z) range.

    Parameters
    ----------
    n : int
        Number of neurons.
    z_min, z_max : float
        Lower and upper z-bounds of the layer (NEST units).

    Returns
    -------
    nest.spatial.free
        NEST spatial position object with shape (n, 3).
    """
    radius = netparams.spatial_radius
    # Polar method: uniform sampling inside a disk without rejection.
    r = radius * np.sqrt(np.random.uniform(0, 1, n))
    theta = np.random.uniform(0, 2 * np.pi, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(z_min, z_max, n)
    return nest.spatial.free(pos=np.column_stack([x, y, z]).tolist())


def _make_syn_spec(weight, delay, decay_beta):
    """
    Build a per-pathway synapse spec dict for nest.Connect.

    When `spatial_weight == 1`:
        weight = base * exp(-d / decay_beta)
    When `spatial_weight == 0`:
        weight = base (flat scalar)

    The base value at d=0 is the same in both modes; toggling
    `spatial_weight` only affects how weight falls off with distance.
    """
    if netparams.args['spatial_weight'] == 1:
        return {
            'synapse_model': 'static_synapse',
            'weight': weight * nest.spatial_distributions.exponential(
                nest.spatial.distance, beta=decay_beta),
            'delay': delay,
        }
    return {
        'synapse_model': 'static_synapse',
        'weight': weight,
        'delay': delay,
    }


def _make_conn_dict(beta, pmax, pflat):
    """
    Build a per-pathway connection dict for nest.Connect.

    When `spatial_sparsity == 1`:
        p(d) = pmax * exp(-d / beta)
    When `spatial_sparsity == 0`:
        p = pflat (flat fallback)
    """
    if netparams.spatial_sparsity:
        return {
            'rule': 'pairwise_bernoulli',
            'p': pmax * nest.spatial_distributions.exponential(
                nest.spatial.distance, beta=beta),
            'allow_autapses': False,
        }
    return {
        'rule': 'pairwise_bernoulli',
        'p': pflat,
        'allow_autapses': False,
    }


# ======================================================================
# PV-PN connectivity
# ======================================================================

def connect_pv_pn(pv_pop, pn_pop, syn_pv_pn, syn_pn_pv, label=''):
    """
    Wire PV-PN connections with empirical reciprocal overrepresentation.

    Per-pair probabilities derived from paired-recording literature:
        Reciprocal   (PV <-> PN):  20.5%
        Uni PV -> PN:               6.8%
        Uni PN -> PV:               2.7%
        Unconnected:               70.0%

    Implementation
    --------------
    A single random draw is performed per ordered (PV, PN) pair, and
    cumulative thresholds determine which connection types are
    instantiated. Weights are built as 1-D numpy arrays for use with
    one-to-one nest.Connect calls (NEST cannot evaluate spatial
    distributions against explicit ID lists).

    When `spatial_weight == 1`, weights decay exponentially with
    Euclidean distance using the per-pathway base value and beta.

    Parameters
    ----------
    pv_pop, pn_pop : NodeCollection
        PV and PN populations to connect.
    syn_pv_pn, syn_pn_pv : dict
        Synapse spec dicts (built via `_make_syn_spec`).
    label : str
        Identifier used in summary print.
    """
    pv_ids = np.array(pv_pop.tolist())
    pn_ids = np.array(pn_pop.tolist())
    n_pv, n_pn = len(pv_ids), len(pn_ids)

    # One random draw per ordered pair
    rolls = np.random.random((n_pv, n_pn))

    # Cumulative probability thresholds
    p_recip = 0.205
    p_uni_inh = p_recip + 0.068
    p_uni_exc = p_uni_inh + 0.027

    recip_mask = rolls < p_recip
    uni_inh_mask = (rolls >= p_recip) & (rolls < p_uni_inh)
    uni_exc_mask = (rolls >= p_uni_inh) & (rolls < p_uni_exc)

    # PV -> PN: reciprocal pairs + unidirectional PV->PN
    # PN -> PV: reciprocal pairs + unidirectional PN->PV
    pv_to_pn_mask = recip_mask | uni_inh_mask
    pn_to_pv_mask = recip_mask | uni_exc_mask

    pv_idx, pn_idx = np.where(pv_to_pn_mask)
    pn_idx2, pv_idx2 = np.where(pn_to_pv_mask.T)

    pv_to_pn_sources = pv_ids[pv_idx].tolist()
    pv_to_pn_targets = pn_ids[pn_idx].tolist()
    pn_to_pv_sources = pn_ids[pn_idx2].tolist()
    pn_to_pv_targets = pv_ids[pv_idx2].tolist()

    n_pv_pn = len(pv_to_pn_sources)
    n_pn_pv = len(pn_to_pv_sources)

    # Per-connection weight arrays
    if netparams.args['spatial_weight'] == 1:
        pv_positions = np.array([nest.GetPosition(n) for n in pv_pop])
        pn_positions = np.array([nest.GetPosition(n) for n in pn_pop])
        d_pv_pn = np.linalg.norm(
            pv_positions[pv_idx] - pn_positions[pn_idx], axis=1)
        d_pn_pv = np.linalg.norm(
            pn_positions[pn_idx2] - pv_positions[pv_idx2], axis=1)
        pv_pn_weights = (netparams.pv_pn_weight
                         * np.exp(-d_pv_pn / netparams.pv_pn_decay))
        pn_pv_weights = (netparams.pn_pv_weight
                         * np.exp(-d_pn_pv / netparams.pn_pv_decay))
    else:
        pv_pn_weights = np.full(n_pv_pn, netparams.pv_pn_weight)
        pn_pv_weights = np.full(n_pn_pv, netparams.pn_pv_weight)

    syn_pv_pn_conn = {
        'synapse_model': syn_pv_pn['synapse_model'],
        'weight': pv_pn_weights,
        'delay': np.full(n_pv_pn, syn_pv_pn['delay']),
    }
    syn_pn_pv_conn = {
        'synapse_model': syn_pn_pv['synapse_model'],
        'weight': pn_pv_weights,
        'delay': np.full(n_pn_pv, syn_pn_pv['delay']),
    }

    if pv_to_pn_sources:
        nest.Connect(pv_to_pn_sources, pv_to_pn_targets,
                     'one_to_one', syn_pv_pn_conn)
    if pn_to_pv_sources:
        nest.Connect(pn_to_pv_sources, pn_to_pv_targets,
                     'one_to_one', syn_pn_pv_conn)

    # Summary print
    n_pairs = n_pv * n_pn
    n_recip = int(recip_mask.sum())
    n_uni_inh = int(uni_inh_mask.sum())
    n_uni_exc = int(uni_exc_mask.sum())
    n_none = n_pairs - n_recip - n_uni_inh - n_uni_exc
    print(
        f'\n  PV-PN connectivity [{label}]:'
        f'\n    Total pairs:           {n_pairs}'
        f'\n    Reciprocal:            {n_recip} '
        f'({100 * n_recip / n_pairs:.1f}%)'
        f'\n    Unidirectional PV->PN: {n_uni_inh} '
        f'({100 * n_uni_inh / n_pairs:.1f}%)'
        f'\n    Unidirectional PN->PV: {n_uni_exc} '
        f'({100 * n_uni_exc / n_pairs:.1f}%)'
        f'\n    No connection:         {n_none} '
        f'({100 * n_none / n_pairs:.1f}%)'
        f'\n    Total PV->PN synapses: {n_pv_pn}'
        f'\n    Total PN->PV synapses: {n_pn_pv}'
    )


# ======================================================================
# PV-PV pair-based connectivity
# ======================================================================

def connect_pv_pv(pv_pop, pv_positions):
    """
    Wire PV-PV pairs using the two-stage model.

    Stage 1 - Connectivity gate (distance-dependent):
        p_connect(d) = p_connect_at_d0 * exp(-d / beta_conn)
        where p_connect_at_d0 = 0.826 (19/23 pairs connected at ~39 um).
        When `spatial_sparsity == 0`, p_connect = p_connect_at_d0 (flat).

    Stage 2 - Conditional category assignment (distance-independent):
        If connected, sample motif from Galarreta & Hestrin Table 1
        renormalised over connected categories:
            gap + reciprocal chem:     10/19 = 52.6%
            gap + unidirectional chem:  3/19 = 15.8%
            gap only:                   1/19 =  5.3%
            reciprocal chem only:       0/19 =  0.0%
            unidirectional chem only:   5/19 = 26.3%

    This preserves:
      - Exact pair-type proportions from Table 1 (regardless of distance)
      - Reciprocal vs unidirectional directionality
      - Gap-chemical co-regulation (0% reciprocal-chem without gap)
    While adding:
      - Spatial sparsity (fewer connections at larger distances)
      - Without distorting category ratios or creating artificial hubs

    Returns
    -------
    pair_types : dict
        Per-pair masks and metadata for post-hoc connectivity profiling
        (used by `classify_pv_by_pair_profile`).
    """
    # --- Unconditional probabilities (Table 1) ---
    p0_gap_recip = netparams.pv_pv_p_gap_recip_chem    # 0.43
    p0_gap_uni = netparams.pv_pv_p_gap_uni_chem        # 0.13
    p0_gap_only = netparams.pv_pv_p_gap_only           # 0.04
    p0_recip_only = netparams.pv_pv_p_recip_chem_only  # 0.00
    p0_uni_only = netparams.pv_pv_p_uni_chem_only      # 0.22

    # Probability of any connection at d=0 (Table 1: 19/23 = 0.826)
    p_connect_at_d0 = (p0_gap_recip + p0_gap_uni + p0_gap_only
                       + p0_recip_only + p0_uni_only)

    # --- Conditional probabilities (renormalised over connected) ---
    cond_gap_recip = p0_gap_recip / p_connect_at_d0    # 0.524
    cond_gap_uni = p0_gap_uni / p_connect_at_d0        # 0.159
    cond_gap_only = p0_gap_only / p_connect_at_d0      # 0.049
    cond_recip_only = p0_recip_only / p_connect_at_d0  # 0.000
    cond_uni_only = p0_uni_only / p_connect_at_d0      # 0.268

    # Cumulative thresholds for Stage 2 category assignment
    c1 = cond_gap_recip
    c2 = c1 + cond_gap_uni
    c3 = c2 + cond_gap_only
    c4 = c3 + cond_recip_only
    c5 = c4 + cond_uni_only  # ~ 1.0

    pv_ids = np.array(pv_pop.tolist())
    n_pv = len(pv_ids)

    # Pairwise distance matrix
    dists = squareform(pdist(pv_positions))
    upper = np.triu(np.ones((n_pv, n_pv), dtype=bool), k=1)

    # ------------------------------------------------------------------
    # Stage 1: connectivity gate
    # ------------------------------------------------------------------
    if netparams.spatial_sparsity:
        beta_conn = netparams.pv_pv_conn_beta
        p_connect = p_connect_at_d0 * np.exp(-dists / beta_conn)
    else:
        p_connect = np.full((n_pv, n_pv), p_connect_at_d0)

    rolls_connect = np.random.random((n_pv, n_pv))
    connected_mask = upper & (rolls_connect < p_connect)

    # ------------------------------------------------------------------
    # Stage 2: conditional category assignment (distance-independent)
    # ------------------------------------------------------------------
    rolls_type = np.random.random((n_pv, n_pv))

    gap_recip_chem_mask = connected_mask & (rolls_type < c1)
    gap_uni_chem_mask = (connected_mask
                        & (rolls_type >= c1) & (rolls_type < c2))
    gap_only_mask = (connected_mask
                     & (rolls_type >= c2) & (rolls_type < c3))
    recip_chem_only_mask = (connected_mask
                            & (rolls_type >= c3) & (rolls_type < c4))
    uni_chem_only_mask = (connected_mask
                          & (rolls_type >= c4) & (rolls_type < c5))

    # --- Determine chemical synapse directions ---
    recip_mask = gap_recip_chem_mask | recip_chem_only_mask
    uni_mask = gap_uni_chem_mask | uni_chem_only_mask
    direction_flip = np.random.random((n_pv, n_pv)) < 0.5

    gap_mask = gap_recip_chem_mask | gap_uni_chem_mask | gap_only_mask

    # --- Build chemical synapse source/target arrays ---
    # Reciprocal: both directions per pair (i->j AND j->i)
    ri, rj = np.where(recip_mask)
    recip_src = np.concatenate([pv_ids[ri], pv_ids[rj]])
    recip_tgt = np.concatenate([pv_ids[rj], pv_ids[ri]])

    # Unidirectional: one direction per pair, randomised
    ui, uj = np.where(uni_mask)
    if len(ui) > 0:
        flips = direction_flip[ui, uj]
        uni_src = np.where(flips, pv_ids[ui], pv_ids[uj])
        uni_tgt = np.where(flips, pv_ids[uj], pv_ids[ui])
    else:
        uni_src = np.array([], dtype=int)
        uni_tgt = np.array([], dtype=int)

    c_src = np.concatenate([recip_src, uni_src])
    c_tgt = np.concatenate([recip_tgt, uni_tgt])
    n_chem = len(c_src)

    # --- Wire chemical synapses ---
    if n_chem > 0:
        if netparams.args['spatial_weight'] == 1:
            d_recip = np.linalg.norm(
                pv_positions[ri] - pv_positions[rj], axis=1)
            w_recip = np.concatenate([
                netparams.pv_pv_weight
                * np.exp(-d_recip / netparams.pv_pv_decay),
                netparams.pv_pv_weight
                * np.exp(-d_recip / netparams.pv_pv_decay),
            ])
            if len(ui) > 0:
                d_uni = np.linalg.norm(
                    pv_positions[ui] - pv_positions[uj], axis=1)
                w_uni = (netparams.pv_pv_weight
                         * np.exp(-d_uni / netparams.pv_pv_decay))
            else:
                w_uni = np.array([])
            weights = np.concatenate([w_recip, w_uni])
        else:
            weights = np.full(n_chem, netparams.pv_pv_weight)

        nest.Connect(
            c_src.tolist(), c_tgt.tolist(),
            'one_to_one',
            {
                'synapse_model': 'static_synapse',
                'weight': weights,
                'delay': np.full(n_chem, netparams.pv_pv_delay),
            },
        )

    # --- Wire gap junctions ---
    n_gap_pairs = 0
    if netparams.gap_junctions_enabled == 1:
        gi, gj = np.where(gap_mask)
        n_gap_pairs = len(gi)
        if n_gap_pairs > 0:
            g_src = np.concatenate([pv_ids[gi], pv_ids[gj]])
            g_tgt = np.concatenate([pv_ids[gj], pv_ids[gi]])
            nest.Connect(
                g_src.tolist(), g_tgt.tolist(),
                'one_to_one',
                {
                    'synapse_model': 'gap_junction',
                    'weight': np.full(len(g_src), netparams.pv_gap_weight),
                },
            )

    # --- Summary print ---
    _print_pv_pv_summary(
        n_pv, dists, connected_mask,
        gap_recip_chem_mask, gap_uni_chem_mask, gap_only_mask,
        recip_chem_only_mask, uni_chem_only_mask,
        recip_mask, uni_mask, n_chem, n_gap_pairs,
    )

    return {
        'pv_ids': pv_ids,
        'gap_recip_chem_mask': gap_recip_chem_mask,
        'gap_uni_chem_mask': gap_uni_chem_mask,
        'gap_only_mask': gap_only_mask,
        'recip_chem_only_mask': recip_chem_only_mask,
        'uni_chem_only_mask': uni_chem_only_mask,
        'uni_direction_flip': direction_flip,
        'connected_mask': connected_mask,
        'dists': dists,
    }


def _print_pv_pv_summary(n_pv, dists, connected_mask,
                         gap_recip_chem_mask, gap_uni_chem_mask,
                         gap_only_mask, recip_chem_only_mask,
                         uni_chem_only_mask, recip_mask, uni_mask,
                         n_chem, n_gap_pairs):
    """Print a summary table of PV-PV connectivity statistics."""
    n_pairs = n_pv * (n_pv - 1) // 2
    n_connected = int(connected_mask.sum())
    n_gap_recip = int(gap_recip_chem_mask.sum())
    n_gap_uni = int(gap_uni_chem_mask.sum())
    n_gap_o = int(gap_only_mask.sum())
    n_recip_o = int(recip_chem_only_mask.sum())
    n_uni_o = int(uni_chem_only_mask.sum())
    n_neither = n_pairs - n_connected
    n_recip_dir = int(recip_mask.sum()) * 2
    n_uni_dir = int(uni_mask.sum())

    def _mean_d(mask):
        ii, jj = np.where(mask)
        return dists[ii, jj].mean() if len(ii) > 0 else 0.0

    cond_label = (
        f'({100 * n_gap_recip / max(n_connected, 1):.0f}/'
        f'{100 * n_gap_uni / max(n_connected, 1):.0f}/'
        f'{100 * n_gap_o / max(n_connected, 1):.0f}/'
        f'{100 * n_recip_o / max(n_connected, 1):.0f}/'
        f'{100 * n_uni_o / max(n_connected, 1):.0f}%)'
    )

    spatial_label = (
        f'ON (beta={netparams.pv_pv_conn_beta})'
        if netparams.spatial_sparsity else 'OFF (flat)'
    )

    print(
        f'\n  PV-PV connectivity (two-stage model):'
        f'\n    Total unordered pairs:        {n_pairs}'
        f'\n    Connected (Stage 1):          {n_connected}  '
        f'({100 * n_connected / n_pairs:.1f}%)'
        f'\n    Neither (unconnected):        {n_neither}  '
        f'({100 * n_neither / n_pairs:.1f}%)'
        f'\n    -- Conditional categories (Stage 2) {cond_label} --'
        f'\n    Gap + reciprocal chem:        {n_gap_recip}  '
        f'mean d={_mean_d(gap_recip_chem_mask):.2f}'
        f'\n    Gap + unidirectional chem:    {n_gap_uni}  '
        f'mean d={_mean_d(gap_uni_chem_mask):.2f}'
        f'\n    Gap only:                     {n_gap_o}  '
        f'mean d={_mean_d(gap_only_mask):.2f}'
        f'\n    Reciprocal chem only:         {n_recip_o}'
        f'\n    Unidirectional chem only:     {n_uni_o}  '
        f'mean d={_mean_d(uni_chem_only_mask):.2f}'
        f'\n    Total directed chem synapses: {n_chem} '
        f'({n_recip_dir} reciprocal + {n_uni_dir} uni)'
        f'\n    Gap junction pairs:           {n_gap_pairs}'
        f'\n    Spatial: {spatial_label}'
    )


def classify_pv_by_pair_profile(pair_types):
    """
    Build per-neuron PV connectivity profiles in 3D space from
    pair-level data.

    Three axes per neuron:
        n_uni_chem  : partners with unidirectional chemical
                      (sent or received)
        n_bidi_chem : partners with bidirectional/reciprocal chemical
        n_gap       : partners with gap junctions (any category)

    Expected clustering pattern (from Galarreta & Hestrin Table 1):
        - Dense cloud at high bidi_chem + high gap (hub neurons)
        - Cluster at moderate uni_chem, low gap (uni-only)
        - Sparse tail near origin (weakly connected)
        - bidi_chem without gap should be rare/absent (0% in Table 1)

    Parameters
    ----------
    pair_types : dict
        Output of `connect_pv_pv`.

    Returns
    -------
    profiles : dict
        Mapping {neuron_id: {n_uni_chem, n_bidi_chem, n_gap,
                             n_neither, total_connected}}.
    """
    pv_ids = pair_types['pv_ids']
    n_pv = len(pv_ids)

    # Symmetrise upper-triangle masks so pair {i,j} counts for both neurons
    gap_recip = pair_types['gap_recip_chem_mask']
    gap_uni = pair_types['gap_uni_chem_mask']
    gap_only = pair_types['gap_only_mask']
    recip_only = pair_types['recip_chem_only_mask']
    uni_only = pair_types['uni_chem_only_mask']

    bidi_mask = gap_recip | gap_recip.T | recip_only | recip_only.T
    uni_mask = gap_uni | gap_uni.T | uni_only | uni_only.T
    gap_mask = (gap_recip | gap_recip.T
                | gap_uni | gap_uni.T
                | gap_only | gap_only.T)

    profiles = {}
    for i, nid in enumerate(pv_ids):
        n_bidi = int(bidi_mask[i].sum())
        n_uni = int(uni_mask[i].sum())
        n_gap = int(gap_mask[i].sum())
        any_conn = bidi_mask[i] | uni_mask[i] | gap_mask[i]
        total_connected = int(any_conn.sum())

        profiles[int(nid)] = {
            'n_uni_chem': n_uni,
            'n_bidi_chem': n_bidi,
            'n_gap': n_gap,
            'n_neither': (n_pv - 1) - total_connected,
            'total_connected': total_connected,
        }

    # Summary statistics
    all_bidi = [p['n_bidi_chem'] for p in profiles.values()]
    all_uni = [p['n_uni_chem'] for p in profiles.values()]
    all_gap = [p['n_gap'] for p in profiles.values()]
    all_total = [p['total_connected'] for p in profiles.values()]

    print(
        f'\n  PV neuron connectivity profiles '
        f'(3-axis: uni_chem x bidi_chem x gap):'
        f'\n    Reciprocal chem partners: '
        f'{np.mean(all_bidi):.1f} +/- {np.std(all_bidi):.1f}'
        f'\n    Unidirectional chem:      '
        f'{np.mean(all_uni):.1f} +/- {np.std(all_uni):.1f}'
        f'\n    Gap junction partners:    '
        f'{np.mean(all_gap):.1f} +/- {np.std(all_gap):.1f}'
        f'\n    Total connected:          '
        f'{np.mean(all_total):.1f} +/- {np.std(all_total):.1f}'
    )

    # Identify hub neurons (top 10% by total connectivity)
    n_hubs = max(1, n_pv // 10)
    sorted_by_total = sorted(profiles.items(),
                             key=lambda x: -x[1]['total_connected'])
    hub_ids = [int(k) for k, _ in sorted_by_total[:n_hubs]]
    hub_bidi = np.mean([profiles[h]['n_bidi_chem'] for h in hub_ids])
    hub_gap = np.mean([profiles[h]['n_gap'] for h in hub_ids])
    print(
        f'    Hub neurons (top {n_hubs}, highest total):'
        f'\n      Mean reciprocal chem: {hub_bidi:.1f}'
        f'\n      Mean gap junctions:   {hub_gap:.1f}'
    )

    return profiles


# ======================================================================
# Main network class
# ======================================================================

class create_pfc_network:
    """
    Build the multilayer PFC spatial network in NEST.

    Construction stages
    -------------------
    1. Build neuron parameter dicts
    2. Create neuron populations (with spatial positions)
    3. Extract and store neuron positions
    4. Create spike recorders and multimeters
    5. Connect chemical synapses (cross-population + recurrent)
    6. Connect PV <-> PN synapses (reciprocal-overrepresented)
    7. Connect PV-PV synapses (6-category pair-based: chem + gap)
    8. Connect recorders to populations
    9. Wire Poisson background input
    """

    def __init__(self):
        # Bookkeeping
        self.senders = []
        self.spiketimes = []
        self.saved_spiketimes = []
        self.saved_senders = []
        self.count = 0
        self.time_window = 50  # 50 * 0.1 ms = 5 ms

        # Build network in logical stages
        self._build_neuron_params()
        self._create_populations()
        self._extract_positions()
        self._create_recorders()
        self._connect_chemical_synapses()
        self._connect_pv_pn_synapses()
        self._connect_pv_pv_synapses()
        self._connect_recorders()
        self._connect_poisson_input()

    # ------------------------------------------------------------------
    # Stage 1 - Neuron parameter dictionaries
    # ------------------------------------------------------------------

    def _build_neuron_params(self):
        """
        Populate self.*_neuronparams dicts for each cell type.

        AdEx parameters chosen to reproduce characteristic firing
        regimes:
            SST: regular-spiking, tonic
            PV : fast-spiking with minimal adaptation
            PN L4: regular-spiking with strong adaptation (bursting)
            PN L5: regular-spiking with PV feedback (PING target)
        """
        syn_tc = {
            'tau_syn_rise_E': netparams.tau_syn_e_rise,
            'tau_syn_decay_E': netparams.tau_syn_e_decay,
            'tau_syn_rise_I': netparams.tau_syn_i_rise,
            'tau_syn_decay_I': netparams.tau_syn_i_decay,
        }

        # SST interneurons - regular-spiking, tonic
        # I_e = 60 pA mean: minimal DC; SST fires from PN synaptic drive.
        self.tonic_neuronparams = {
            'C_m': 120.,
            'g_L': 8.,
            'E_L': -65.,
            'V_th': -52.,
            'Delta_T': 2.,
            'tau_w': 150.,
            'a': 3.0,
            'b': 60.,
            'V_reset': -65.,
            'V_peak': 0.,
            'I_e': nest.random.normal(mean=60., std=5.),
            't_ref': 2.,
            'V_m': -65.,
            'tau_syn_rise_I': 0.5,
            'tau_syn_decay_I': 5.0,
            'tau_syn_rise_E': 0.5,
            'tau_syn_decay_E': 4.0,
        }

        # PV interneurons - fast-spiking
        # I_e mean = 30 pA: PV fire when driven by PN spikes.
        # tau_syn_decay_I = 2.0 ms: fast GABA-A onto PV (Geiger et al.
        # 1997 - PV->PV synapses have fast kinetics; slower GABA here
        # extended the inhibitory window and locked cells into a slow
        # volley rhythm).
        self.tonic_fast_neuronparams = {
            'C_m': 70.,
            'g_L': 14.,
            'E_L': -65.,
            'V_th': -52.,
            'Delta_T': 1.,
            'tau_w': 10.,
            'a': 0.,
            'b': 0.,
            'V_reset': -65.,
            'V_peak': 0.,
            'I_e': nest.random.normal(mean=30., std=5.),
            't_ref': 1.,
            'V_m': -65.,
            'tau_syn_rise_E': 0.2,
            'tau_syn_decay_E': 1.0,   # Geiger et al. 1997: ~1-2 ms
            'tau_syn_rise_I': 0.2,
            'tau_syn_decay_I': 2.0,   # Bartos et al. 2002: PV->PV ~1.5-2 ms
        }

        # PN L4 - regular-spiking pyramidal neurons
        # I_e mean = 200 pA: above rheobase so PNs fire tonically and
        # PV inhibition modulates rate rather than silencing them.
        # tau_syn_decay_I = 5 ms: perisomatic GABA-A clears in ~10 ms,
        # leaving a recovery window per gamma cycle.
        self.pn_l4_params = {
            'C_m': 180.,
            'g_L': 8.,
            'E_L': -70.,
            'V_th': -50.,
            'Delta_T': 2.5,
            'tau_w': 80.,
            'a': 1.0,
            'b': 40.,
            'V_reset': -55.,
            'V_peak': 0.,
            'I_e': nest.random.normal(mean=200., std=10.),
            't_ref': nest.random.normal(
                mean=5.0, std=0.2),
            'V_m': nest.random.normal(
                mean=-60.0, std=10.0),
            'tau_syn_rise_I': 0.5,
            'tau_syn_decay_I': 5.,
            'tau_syn_rise_E': 0.5,
            'tau_syn_decay_E': 2.0,
        }

        # Bursting pyramidal neurons (legacy, smaller subset in L5)
        self.bursting_neuronparams = {
            'C_m': 200.,
            'g_L': 10.,
            'E_L': -58.,
            'V_th': -50.,
            'Delta_T': 2.,
            'tau_w': 120.,
            'a': 2.,
            'b': 100.,
            'V_reset': -46.,
            'I_e': 40.,
            't_ref': nest.random.normal(
                mean=netparams.t_ref_bursting_mean,
                std=netparams.t_ref_bursting_std),
            'V_m': nest.random.normal(
                mean=netparams.V_m_mean, std=netparams.V_m_std),
            **syn_tc,
        }

        # PN L5 - regular-spiking pyramidal neurons with PV feedback
        # I_e mean = 270 pA: rheobase ~60 pA, excess drive ~200 pA;
        # time-to-threshold fits inside the recovery window after a
        # PV burst (2 spikes x ~3.5 ms decay).
        # std = 5 adds heterogeneity so neurons fire at different
        # phases and the population rate never simultaneously drops
        # to zero.
        # tau_syn_decay_I = 2.0 ms: fast perisomatic GABA-A.
        self.pn_l5_params = {
            'C_m': 200.,
            'g_L': 8.,
            'E_L': -70.,
            'V_th': -52.,
            'Delta_T': 2.5,
            'tau_w': 150.,
            'a': 1.0,
            'b': 80.,
            'V_reset': -56.,
            'V_peak': 20.,
            'I_e': nest.random.normal(mean=270., std=5.),
            't_ref': 2.0,
            'V_m': nest.random.normal(mean=-60.0, std=3.0),
            'tau_syn_rise_E': 0.5,
            'tau_syn_decay_E': 2.0,   # Hestrin 1993
            'tau_syn_rise_I': 0.5,
            'tau_syn_decay_I': 2.0,   # Hefft & Jonas 2005
        }

        # Legacy alias retained for backwards compatibility
        self.rg_spiking_neuronparams = self.pn_l5_params

    # ------------------------------------------------------------------
    # Stage 2 - Create neuron populations
    # ------------------------------------------------------------------

    def _create_populations(self):
        """
        Create all NEST neuron populations with 3D spatial positions.

        Coordinate system (1 NEST unit = 200 um):
            x, y : horizontal plane - circular disk of radius
                   `spatial_radius`
            z    : cortical depth - L4 above (z > 0), L5 below (z <= 0)
        """
        MODEL = 'aeif_cond_beta_nestml'

        # L4 populations
        self.pn_exc_l4 = nest.Create(
            MODEL, netparams.pn_l4_count,
            self.pn_l4_params,
            positions=_make_layer_positions(
                netparams.pn_l4_count,
                netparams.l4_z_min, netparams.l4_z_max),
        )

        self.sst_inh_l4 = nest.Create(
            MODEL, netparams.sst_inh_l4_count,
            self.tonic_neuronparams,
            positions=_make_layer_positions(
                netparams.sst_inh_l4_count,
                netparams.l4_z_min, netparams.l4_z_max),
        )

        # L5 populations
        self.pn_exc_l5 = nest.Create(
            MODEL, netparams.pn_exc_l5_count,
            self.pn_l5_params,
            positions=_make_layer_positions(
                netparams.pn_exc_l5_count,
                netparams.l5_z_min, netparams.l5_z_max),
        )

        self.pv_inh_l5 = nest.Create(
            MODEL, netparams.total_pv,
            self.tonic_fast_neuronparams,
            positions=_make_layer_positions(
                netparams.total_pv,
                netparams.l5_z_min, netparams.l5_z_max),
        )

    # ------------------------------------------------------------------
    # Stage 3 - Extract neuron positions
    # ------------------------------------------------------------------

    def _extract_positions(self):
        """Cache (n, 3) position arrays for every population."""
        self.pn_exc_l4_positions = np.array(
            [nest.GetPosition(n) for n in self.pn_exc_l4])
        self.pn_exc_l5_positions = np.array(
            [nest.GetPosition(n) for n in self.pn_exc_l5])
        self.sst_inh_l4_positions = np.array(
            [nest.GetPosition(n) for n in self.sst_inh_l4])
        self.pv_inh_l5_positions = np.array(
            [nest.GetPosition(n) for n in self.pv_inh_l5])

    # ------------------------------------------------------------------
    # Stage 4 - Create spike recorders and multimeters
    # ------------------------------------------------------------------

    def _create_recorders(self):
        """Instantiate one spike recorder and one multimeter per
        population.
        """
        self.spike_detector_pn_exc_l4 = nest.Create(
            'spike_recorder', netparams.pn_l4_count)
        self.spike_detector_pn_exc_l5 = nest.Create(
            'spike_recorder', netparams.pn_exc_l5_count)
        self.spike_detector_sst_inh_l4 = nest.Create(
            'spike_recorder', netparams.sst_inh_l4_count)
        self.spike_detector_pv_inh_l5 = nest.Create(
            'spike_recorder', netparams.total_pv)

        self.mm_pn_exc_l4 = nest.Create('multimeter', netparams.mm_params)
        self.mm_pn_exc_l5 = nest.Create('multimeter', netparams.mm_params)
        self.mm_sst_inh_l4 = nest.Create('multimeter', netparams.mm_params)
        self.mm_pv_inh_l5 = nest.Create('multimeter', netparams.mm_params)

    # ------------------------------------------------------------------
    # Stage 5 - Chemical synapses
    # ------------------------------------------------------------------

    def _connect_chemical_synapses(self):
        """
        Wire chemical (static) synapses with per-pathway spatial
        connectivity.

        Each connection is specified as:
            (source_key, target_key, conn_dict, syn_spec)

        conn_dict is built per-pathway via `_make_conn_dict` using the
        biologically motivated decay constants and peak probabilities
        defined in `parameters.py`. When `spatial_sparsity` is off, the
        flat fallback probability is used instead.

        PV <-> PN connections are handled separately in
        `_connect_pv_pn_synapses` because they use the
        reciprocal-overrepresentation scheme.
        """
        populations = {
            'pn_exc_l4': self.pn_exc_l4,
            'pn_exc_l5': self.pn_exc_l5,
            'sst_inh_l4': self.sst_inh_l4,
            'pv_inh_l5': self.pv_inh_l5,
        }

        def cd(beta, pmax, pflat):
            return _make_conn_dict(beta, pmax, pflat)

        def syn(weight, delay, decay):
            return _make_syn_spec(weight, delay, decay)

        # Cross-population connections
        cross_connections = [
            # Within L4: PN <-> SST
            ('pn_exc_l4', 'sst_inh_l4',
             cd(netparams.pn_sst_l4_beta,
                netparams.pn_sst_l4_pmax,
                netparams.pn_sst_l4_pflat),
             syn(netparams.pn_sst_weight,
                 netparams.pn_sst_delay,
                 netparams.pn_sst_decay)),

            ('sst_inh_l4', 'pn_exc_l4',
             cd(netparams.sst_pn_l4_beta,
                netparams.sst_pn_l4_pmax,
                netparams.sst_pn_l4_pflat),
             syn(netparams.sst_pn_weight,
                 netparams.sst_pn_delay,
                 netparams.sst_pn_decay)),

            # Cross-layer feedforward: L4 PN -> L5 PN
            ('pn_exc_l4', 'pn_exc_l5',
             cd(netparams.pn_l4_pn_l5_ff_beta,
                netparams.pn_l4_pn_l5_ff_pmax,
                netparams.pn_l4_pn_l5_ff_pflat),
             syn(netparams.pn_pn_weight,
                 netparams.pn_pn_delay,
                 netparams.pn_pn_decay)),
        ]

        # Recurrent connections
        self_connections = [
            ('pn_exc_l4', 'pn_exc_l4',
             cd(netparams.pn_l4_self_beta,
                netparams.pn_l4_self_pmax,
                netparams.pn_l4_self_pflat),
             syn(netparams.pn_pn_weight,
                 netparams.pn_pn_delay,
                 netparams.pn_pn_decay)),

            ('pn_exc_l5', 'pn_exc_l5',
             cd(netparams.pn_l5_self_beta,
                netparams.pn_l5_self_pmax,
                netparams.pn_l5_self_pflat),
             syn(netparams.pn_pn_weight,
                 netparams.pn_pn_delay,
                 netparams.pn_pn_decay)),

            # Bidirectional cross-layer recurrence (currently disabled
            # via pmax = 0 in parameters.py)
            ('pn_exc_l4', 'pn_exc_l5',
             cd(netparams.pn_l4_pn_l5_rec_beta,
                netparams.pn_l4_pn_l5_rec_pmax,
                netparams.pn_l4_pn_l5_rec_pflat),
             syn(netparams.pn_pn_weight,
                 netparams.pn_pn_delay,
                 netparams.pn_pn_decay)),

            ('pn_exc_l5', 'pn_exc_l4',
             cd(netparams.pn_l5_pn_l4_rec_beta,
                netparams.pn_l5_pn_l4_rec_pmax,
                netparams.pn_l5_pn_l4_rec_pflat),
             syn(netparams.pn_pn_weight,
                 netparams.pn_pn_delay,
                 netparams.pn_pn_decay)),
            # NOTE: PV-PV chemical synapses are wired pair-by-pair in
            # _connect_pv_pv_synapses() using the empirical joint
            # distribution, not via the bulk Bernoulli scheme above.
        ]

        self.cross_couplings = {}
        self.self_couplings = {}

        def _wire(connections, store_dict):
            for src, tgt, conn_dict, syn_spec in connections:
                store_dict[f'{src}_{tgt}'] = nest.Connect(
                    populations[src], populations[tgt],
                    conn_dict, syn_spec,
                )

        _wire(cross_connections, self.cross_couplings)
        _wire(self_connections, self.self_couplings)

    # ------------------------------------------------------------------
    # Stage 6 - PV <-> PN reciprocal connectivity
    # ------------------------------------------------------------------

    def _connect_pv_pn_synapses(self):
        """Wire PV <-> PN with empirical reciprocal overrepresentation."""
        syn_pv_to_pn = _make_syn_spec(
            netparams.pv_pn_weight,
            netparams.pv_pn_delay,
            netparams.pv_pn_decay)
        syn_pn_to_pv = _make_syn_spec(
            netparams.pn_pv_weight,
            netparams.pn_pv_delay,
            netparams.pn_pv_decay)

        connect_pv_pn(self.pv_inh_l5, self.pn_exc_l5,
                      syn_pv_to_pn, syn_pn_to_pv,
                      label='PV_L5 <-> PN_L5')

    # ------------------------------------------------------------------
    # Stage 7 - PV-PV synapses (chemical + gap junctions)
    # ------------------------------------------------------------------

    def _connect_pv_pv_synapses(self):
        """
        Wire PV-PV pairs using the 6-category joint distribution from
        Galarreta & Hestrin 2002 Table 1, then build per-neuron
        connectivity profiles.
        """
        self.pv_pv_pair_types = connect_pv_pv(
            self.pv_inh_l5, self.pv_inh_l5_positions)
        self.pv_profiles = classify_pv_by_pair_profile(
            self.pv_pv_pair_types)

    # ------------------------------------------------------------------
    # Stage 8 - Connect recorders to populations
    # ------------------------------------------------------------------

    def _connect_recorders(self):
        """
        Attach spike recorders and multimeters to every population.

        Spike recorders are connected one-to-one (one per neuron).
        Each multimeter records from a single sampled neuron only,
        keeping memory usage low.
        """
        # Sampled neuron index for V_m recording
        self.mm_neuron_idx = 60

        recorder_map = [
            (self.pn_exc_l4, self.spike_detector_pn_exc_l4,
             self.mm_pn_exc_l4),
            (self.pn_exc_l5, self.spike_detector_pn_exc_l5,
             self.mm_pn_exc_l5),
            (self.sst_inh_l4, self.spike_detector_sst_inh_l4,
             self.mm_sst_inh_l4),
            (self.pv_inh_l5, self.spike_detector_pv_inh_l5,
             self.mm_pv_inh_l5),
        ]

        for pop, sd, mm in recorder_map:
            nest.Connect(pop, sd, 'one_to_one')
            sd.n_events = 0
            nest.Connect(mm, pop[self.mm_neuron_idx])

    # ------------------------------------------------------------------
    # Stage 9 - Poisson background input
    # ------------------------------------------------------------------

    def _connect_poisson_input(self):
        """
        Wire one independent Poisson generator per neuron for each
        population.

        Using one-to-one (rather than one shared generator) avoids the
        shared-noise synchrony artefact in which all neurons receive
        identical spike trains. Rates represent aggregate
        thalamocortical and corticocortical background bombardment,
        placing neurons in the high-conductance state needed for
        gamma-range E/I dynamics (Brunel 2000; Wang 2010).
        """
        syn_poisson = {
            'synapse_model': 'static_synapse',
            'weight': netparams.poisson_weight,
            'delay': 1.0,
        }

        populations = [
            (self.pn_exc_l4, netparams.pn_l4_count,
             netparams.poisson_rate_pn_l4, 'PN L4'),
            (self.pn_exc_l5, netparams.pn_exc_l5_count,
             netparams.poisson_rate_pn_l5, 'PN L5'),
            (self.sst_inh_l4, netparams.sst_inh_l4_count,
             netparams.poisson_rate_sst, 'SST L4'),
            (self.pv_inh_l5, netparams.total_pv,
             netparams.poisson_rate_pv, 'PV L5'),
        ]

        for pop, n, rate, label in populations:
            pg = nest.Create('poisson_generator', n, {'rate': rate})
            nest.Connect(pg, pop, 'one_to_one', syn_poisson)
            print(f'  [Poisson] {label}: {n} generators @ {rate:.0f} Hz, '
                  f'w={netparams.poisson_weight} nS')

        # Inhibitory Poisson onto L5 PN as a proxy for missing SST/VIP
        # inhibition (currently rate=0; included for future use)
        syn_poisson_inh = {
            'synapse_model': 'static_synapse',
            'weight': netparams.poisson_weight_inh,
            'delay': 1.0,
        }
        n_l5pn = netparams.pn_exc_l5_count
        pg_inh = nest.Create(
            'poisson_generator', n_l5pn,
            {'rate': netparams.poisson_rate_pn_l5_inh})
        nest.Connect(pg_inh, self.pn_exc_l5, 'one_to_one', syn_poisson_inh)
        print(f'  [Poisson] PN L5 inh: {n_l5pn} generators @ '
              f'{netparams.poisson_rate_pn_l5_inh:.0f} Hz, '
              f'w={netparams.poisson_weight_inh} nS')
