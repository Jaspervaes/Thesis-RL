"""
Generate thesis figures from results/all_results.json.

3-step only (1-step / 2-step dropped from the paper). Adds a focused
LSTM-DQN subset comparison chart reading results/lstm_joint_vs_subset.json.

Usage:
    python plot_results.py
    python plot_results.py --results results/all_results.json
    python plot_results.py --out results/figures

Figures produced:
    fig2_absolute_performance.pdf   — grouped bars: avg outcome per method, RCT vs CONF
    fig3_rct_vs_conf.pdf            — small multiples per method: RCT vs CONF gain
    fig5_gain_heatmap.pdf           — heatmap method × {RCT, CONF}, colour = % gain
    fig7_dqn_vs_procause.pdf        — LSTM-DQN baseline vs causal-reward variants
    fig8_confounding_robustness.pdf — confounding gap per method
    fig1_lstm_subsets.pdf           — LSTM-DQN gain by intervention subset (CONF [+RCT])
"""
import sys
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── Style ─────────────────────────────────────────────────────────────────────

# All internal JSON keys, in display order
METHODS = ['kmeans', 'lstm', 'rims', 'multiModelCQL', 'singleModelCQL',
           'procause_lstm', 'procause_econml', 'lstm_dqn_dragonnet', 'lstm_dqn_tabpfn']

# Default display label per internal key (used in all non-causal figures)
METHOD_LABELS = {
    'kmeans':              'K-Means-FQI',
    'lstm':                'LSTM-DQN',
    'rims':                'RIMS-DQN',
    'multiModelCQL':       'CQL-MN',
    'singleModelCQL':      'CQL-SN',
    'procause_lstm':       'LSTM-DQN-SLearner',
    'procause_econml':     'LSTM-DQN-GBR',
    'lstm_dqn_dragonnet':  'LSTM-DQN-DragonNet',
    'lstm_dqn_tabpfn':     'LSTM-DQN-TabPFN',
}

# Causal figure uses full ProCause names for the two S-Learner variants
CAUSAL_LABELS = {
    **METHOD_LABELS,
    'procause_lstm':   'ProCause LSTM-DQN',
    'procause_econml': 'ProCause EconML-DQN',
}

# Color keyed by display name, then mapped to internal keys via METHOD_LABELS
_MC = {
    'LSTM-DQN':            '#1F77B4',
    'CQL-SN':              '#AEC7E8',
    'CQL-MN':              '#6BAED6',
    'K-Means-FQI':         '#2CA02C',
    'LSTM-DQN-GBR':        '#E8735A',
    'LSTM-DQN-SLearner':   '#FF9896',
    'LSTM-DQN-TabPFN':     '#D62728',
    'LSTM-DQN-DragonNet':  '#FF7F0E',
    'RIMS-DQN':            '#9467BD',
}
COLORS = {k: _MC[v] for k, v in METHOD_LABELS.items()}

# Methods shown in every figure except the causal comparison
MAIN_METHODS = ['lstm', 'singleModelCQL', 'multiModelCQL', 'kmeans',
                'lstm_dqn_tabpfn', 'lstm_dqn_dragonnet', 'rims']

# Methods shown only in the causal comparison figure (fig7)
CAUSAL_METHODS = ['lstm', 'procause_lstm', 'procause_econml',
                  'lstm_dqn_dragonnet', 'lstm_dqn_tabpfn']

LSTM_BLUE = '#1F77B4'   # LSTM-DQN color, used for the subset chart

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

STEPS = 3   # only 3-step retained in the paper


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(path):
    with open(path) as f:
        return json.load(f)


def get_agg(results, method, suffix):
    key = f"{method}_{suffix}_{STEPS}"
    if key not in results:
        return None
    return results[key]['aggregated']


def get_gain(agg, bank_key='Bank'):
    """% gain of the method vs Bank, with propagated std."""
    if agg is None:
        return None, None
    bank_m = agg[bank_key]['mean']
    policy_key = [k for k in agg if k not in ('Bank', 'Random')][0]
    pol_m   = agg[policy_key]['mean']
    pol_std = agg[policy_key]['std']
    bank_std = agg[bank_key]['std']
    gain = ((pol_m / bank_m) - 1) * 100 if bank_m > 0 else float('nan')
    if bank_m > 0 and pol_m != 0:
        rel_err = np.sqrt((pol_std / abs(pol_m))**2 + (bank_std / abs(bank_m))**2)
        gain_std = abs(gain) * rel_err
    else:
        gain_std = float('nan')
    return gain, gain_std


def get_per_seed_gains(results, method, suffix):
    key = f"{method}_{suffix}_{STEPS}"
    if key not in results:
        return []
    entry = results[key]
    agg = entry['aggregated']
    per_seed = entry.get('per_seed', {})
    policy_key = [k for k in agg if k not in ('Bank', 'Random')][0]
    gains = []
    for seed_data in per_seed.values():
        bank = seed_data.get('Bank', None)
        pol  = seed_data.get(policy_key, None)
        if bank and pol and bank > 0:
            gains.append(((pol / bank) - 1) * 100)
    return gains


# ── Figure 4: Per-Seed Strip Plot ────────────────────────────────────────────

def fig4_seed_variance(results, out_dir, suffixes):
    """Strip plot (one dot per seed) of % gain vs Bank — 3-step only, MAIN_METHODS."""
    n_suf = len(suffixes)
    fig, axes = plt.subplots(1, n_suf, figsize=(6 * n_suf, 5), sharey=True)
    if n_suf == 1:
        axes = [axes]

    for ax, suffix in zip(axes, suffixes):
        positions, tick_labels = [], []
        for pos, method in enumerate(MAIN_METHODS):
            key = f"{method}_{suffix}_{STEPS}"
            if key not in results:
                continue
            entry = results[key]
            agg   = entry['aggregated']
            bank_m = agg['Bank']['mean']
            if bank_m <= 0:
                continue
            pol_key = next(k for k in agg if k not in ('Bank', 'Random'))
            per_seed = agg[pol_key].get('per_seed', [])
            if not per_seed:
                continue

            gains = [((v - bank_m) / bank_m) * 100 for v in per_seed]
            color = COLORS[method]

            # jitter x slightly so overlapping dots separate
            jitter = (np.random.RandomState(pos).rand(len(gains)) - 0.5) * 0.25
            ax.scatter(gains, [pos] * len(gains) + jitter,
                       color=color, s=60, zorder=3, edgecolors='black', linewidths=0.4)

            # mean marker
            ax.scatter([np.mean(gains)], [pos], marker='|', s=200,
                       color='black', zorder=4, linewidths=2)

            positions.append(pos)
            tick_labels.append(METHOD_LABELS[method])

        ax.axvline(0, color='grey', lw=0.8, linestyle='--', alpha=0.6)
        ax.set_yticks(positions)
        ax.set_yticklabels(tick_labels, fontsize=9)
        ax.set_xlabel('% Gain vs Bank Policy')
        ax.set_title(f'{suffix} Data')
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Per-Seed Distribution of % Gain over Bank Policy (3-step)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_seed_variance.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Figure 2: Absolute Performance Grouped Bar Chart ─────────────────────────

def fig2_absolute_performance(results, out_dir, suffixes):
    """One bar per method per suffix (RCT, CONF) — 3-step only."""
    available_methods = [m for m in MAIN_METHODS if any(get_agg(results, m, s) for s in suffixes)]
    n_methods = len(available_methods)

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * n_methods), 6))

    width = 0.4
    x = np.arange(n_methods)

    # Baselines (use first suffix's first method as reference)
    ref_agg = next((get_agg(results, m, s) for m in available_methods for s in suffixes
                    if get_agg(results, m, s)), None)
    if ref_agg:
        ax.axhline(ref_agg['Bank']['mean'], color='black', lw=1.5, linestyle='--',
                   label='Bank policy', zorder=5)
        if 'Random' in ref_agg:
            ax.axhline(ref_agg['Random']['mean'], color='grey', lw=1.2, linestyle=':',
                       label='Random policy', zorder=5)

    for si, suffix in enumerate(suffixes):
        means, errs, bar_colors = [], [], []
        for method in available_methods:
            agg = get_agg(results, method, suffix)
            if agg is None:
                means.append(0); errs.append(0)
            else:
                pk = [k for k in agg if k not in ('Bank', 'Random')][0]
                means.append(agg[pk]['mean'])
                errs.append(agg[pk]['std'])
            bar_colors.append(COLORS[method])

        offset = (si - (len(suffixes) - 1) / 2) * width
        alpha = 0.9 if suffix == 'RCT' else 0.55
        hatch = '' if suffix == 'RCT' else '///'
        ax.bar(x + offset, means, width * 0.9, yerr=errs, capsize=3,
               color=bar_colors, alpha=alpha, hatch=hatch,
               edgecolor='black', linewidth=0.5, error_kw={'lw': 1})

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in available_methods],
                       rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Average Outcome')
    ax.set_title('Absolute Performance by Method (3-step)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Custom legend: RCT = solid, CONF = hatched. Anchored outside the bar area.
    legend_elems = [
        mpatches.Patch(facecolor='grey', alpha=0.9, edgecolor='black', label='RCT'),
        mpatches.Patch(facecolor='grey', alpha=0.55, edgecolor='black', hatch='///', label='CONF'),
    ]
    ax.legend(handles=legend_elems, frameon=False, fontsize=13,
              loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_absolute_performance.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Figure 3 (A2): RCT vs CONF per method ────────────────────────────────────

def fig3_rct_vs_conf(results, out_dir):
    """Side-by-side RCT vs CONF gain bar for each method — 3-step only."""
    available = [m for m in MAIN_METHODS
                 if (get_agg(results, m, 'RCT') is not None or
                     get_agg(results, m, 'CONF') is not None)]
    if not available:
        print("[skip] fig3: no data")
        return

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(available)), 6))
    x = np.arange(len(available))
    width = 0.4

    for si, suffix in enumerate(['RCT', 'CONF']):
        gains, errs = [], []
        for method in available:
            g, s = get_gain(get_agg(results, method, suffix))
            gains.append(g if g is not None else 0)
            errs.append(s if (s is not None and not np.isnan(s)) else 0)

        offset = (si - 0.5) * width
        alpha = 0.9 if suffix == 'RCT' else 0.5
        hatch = '' if suffix == 'RCT' else '///'
        bar_colors = [COLORS[m] for m in available]
        ax.bar(x + offset, gains, width * 0.9, yerr=errs, capsize=3,
               color=bar_colors, alpha=alpha, hatch=hatch,
               edgecolor='black', linewidth=0.5, error_kw={'lw': 1.2})

    ax.axhline(0, color='grey', lw=0.8, linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in available],
                       rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('% Gain vs Bank Policy')
    ax.set_title('RCT vs Confounded Data — % Gain over Bank (3-step)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    legend_elems = [
        mpatches.Patch(facecolor='grey', alpha=0.9, edgecolor='black', label='RCT'),
        mpatches.Patch(facecolor='grey', alpha=0.5, edgecolor='black', hatch='///', label='CONF'),
    ]
    ax.legend(handles=legend_elems, frameon=False, fontsize=13,
              loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_rct_vs_conf.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Figure 5 (A3): Gain Heatmap ───────────────────────────────────────────────

def fig5_gain_heatmap(results, out_dir, suffixes):
    """Heatmap: methods × suffixes (RCT, CONF), colour = % gain. 3-step only."""
    methods = MAIN_METHODS
    fig, ax = plt.subplots(figsize=(max(4, 1.3 * len(suffixes)) + 2, 0.5 * len(methods) + 2))
    data = np.full((len(methods), len(suffixes)), np.nan)
    for mi, method in enumerate(methods):
        for si, suffix in enumerate(suffixes):
            g, _ = get_gain(get_agg(results, method, suffix))
            if g is not None:
                data[mi, si] = g

    vmax = np.nanmax(np.abs(data))
    im = ax.imshow(data, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(suffixes)))
    ax.set_xticklabels(suffixes)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_title('% Gain over Bank Policy (3-step)', fontweight='bold')

    for mi in range(len(methods)):
        for si in range(len(suffixes)):
            val = data[mi, si]
            text = f'{val:+.1f}%' if not np.isnan(val) else 'N/A'
            ax.text(si, mi, text, ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if not np.isnan(val) and abs(val) > vmax * 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='% Gain vs Bank', shrink=0.8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig5_gain_heatmap.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Figure 7 (paper Fig 8): Causal-reward variants vs baseline ───────────────

def fig7_dqn_vs_procause(results, out_dir, suffixes):
    """LSTM-DQN baseline vs causal-reward variants — 3-step only."""
    group_methods = [m for m in CAUSAL_METHODS
                     if any(get_agg(results, m, s) is not None for s in suffixes)]
    if not group_methods:
        print("[skip] fig7: no LSTM-DQN family methods")
        return

    n_suf = len(suffixes)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(group_methods)), 5.5))
    x = np.arange(len(group_methods))
    width = 0.4

    # Baselines from any available
    ref_agg = next((get_agg(results, m, s) for m in group_methods for s in suffixes
                    if get_agg(results, m, s)), None)
    if ref_agg:
        ax.axhline(ref_agg['Bank']['mean'], color='black', lw=1.5, linestyle='--',
                   label='Bank policy', zorder=5)
        if 'Random' in ref_agg:
            ax.axhline(ref_agg['Random']['mean'], color='grey', lw=1.2, linestyle=':',
                       label='Random policy', zorder=5)

    for si, suffix in enumerate(suffixes):
        means, errs, bar_colors = [], [], []
        for method in group_methods:
            agg = get_agg(results, method, suffix)
            if agg is None:
                means.append(0); errs.append(0)
            else:
                pk = [k for k in agg if k not in ('Bank', 'Random')][0]
                means.append(agg[pk]['mean'])
                errs.append(agg[pk]['std'])
            bar_colors.append(COLORS[method])

        offset = (si - (n_suf - 1) / 2) * width
        alpha = 0.9 if suffix == 'RCT' else 0.55
        hatch = '' if suffix == 'RCT' else '///'
        ax.bar(x + offset, means, width * 0.9, yerr=errs, capsize=3,
               color=bar_colors, alpha=alpha, hatch=hatch,
               edgecolor='black', linewidth=0.5, error_kw={'lw': 1})

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in group_methods],
                       rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Average Outcome')
    ax.set_title('Causal Reward Estimation: LSTM-DQN Baseline vs Variants (3-step)',
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    legend_elems = [
        mpatches.Patch(facecolor='grey', alpha=0.9, edgecolor='black', label='RCT'),
        mpatches.Patch(facecolor='grey', alpha=0.55, edgecolor='black', hatch='///', label='CONF'),
    ]
    ax.legend(handles=legend_elems, frameon=False, fontsize=13,
              loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig7_dqn_vs_procause.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Figure: Normalized Confounding Gap ───────────────────────────────────────

# Maps JSON aggregated label base → internal method key
_JSON_BASE_TO_KEY = {
    'KMeans':              'kmeans',
    'LSTM':                'lstm',
    'RIMS-DQN':            'rims',
    'CQL-MN':              'multiModelCQL',
    'CQL-SN':              'singleModelCQL',
    'ProCause LSTM-DQN':   'procause_lstm',
    'ProCause EconML-DQN': 'procause_econml',
    'LSTM-DQN-DragonNet':  'lstm_dqn_dragonnet',
    'LSTM-DQN-TabPFN':     'lstm_dqn_tabpfn',
}


def fig_normalized_gap(results, out_dir):
    """Normalized confounding gap with propagated std error bars, computed from results."""
    import math

    bank = next(
        v['aggregated']['Bank']['mean']
        for v in results.values()
        if 'aggregated' in v and 'Bank' in v.get('aggregated', {})
    )

    gains, stds = {}, {}
    for key, val in results.items():
        if not key.endswith('_3'):
            continue
        for k, v in val.get('aggregated', {}).items():
            if k in ('Random', 'Bank'):
                continue
            gains[k] = (v['mean'] - bank) / bank * 100
            stds[k]  = v['std']  / bank * 100

    rows = []
    for k, rct in gains.items():
        if 'RCT' not in k:
            continue
        base = k.replace(' RCT (3-step)', '')
        internal_key = _JSON_BASE_TO_KEY.get(base)
        if internal_key not in MAIN_METHODS:
            continue
        ck = base + ' CONF (3-step)'
        if ck not in gains:
            continue
        conf = gains[ck]
        norm = (rct - conf) / abs(rct) * 100
        sigma = math.sqrt(
            (conf / rct**2)**2 * stds[k]**2 +
            (1    / rct   )**2 * stds[ck]**2
        ) * 100
        rows.append((METHOD_LABELS[internal_key], norm, sigma, COLORS[internal_key]))

    rows.sort(key=lambda r: r[1])

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    sigmas = [r[2] for r in rows]
    colors = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(labels, values, xerr=sigmas, color=colors,
            edgecolor='black', linewidth=0.5, alpha=1.0,
            error_kw={'lw': 1.5, 'capsize': 4, 'capthick': 1.5, 'ecolor': 'black'})

    ax.axvline(0, color='grey', lw=0.8, linestyle='--', alpha=0.6)
    ax.set_xlabel('(RCT gain − CONF gain) / |RCT gain|  [%]')
    ax.set_title('Normalized Confounding Gap by Method (3-step)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig9_normalized_gap.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Figure 8 (paper Fig 7): Confounding Gap ──────────────────────────────────

def fig8_confounding_robustness(results, out_dir):
    """Single-panel bar chart of confounding gap (RCT gain − CONF gain) per method."""
    available = [m for m in MAIN_METHODS
                 if f"{m}_CONF_{STEPS}" in results and f"{m}_RCT_{STEPS}" in results]
    if not available:
        print("[skip] fig8: need both RCT and CONF results")
        return

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(available)), 5))
    gaps, gap_errs, colors_list, labels_list = [], [], [], []
    for method in available:
        g_rct, s_rct   = get_gain(get_agg(results, method, 'RCT'))
        g_conf, s_conf = get_gain(get_agg(results, method, 'CONF'))
        if g_rct is not None and g_conf is not None:
            gaps.append(g_rct - g_conf)
            gap_errs.append(np.sqrt(s_rct**2 + s_conf**2)
                            if (s_rct is not None and s_conf is not None
                                and not np.isnan(s_rct) and not np.isnan(s_conf))
                            else 0)
            colors_list.append(COLORS[method])
            labels_list.append(METHOD_LABELS[method])

    x = np.arange(len(gaps))
    bars = ax.bar(x, gaps, 0.65, yerr=gap_errs, capsize=4,
                  color=colors_list, alpha=0.85,
                  edgecolor='black', linewidth=0.5,
                  error_kw={'lw': 1.2, 'capthick': 1.2, 'ecolor': 'black'})
    ax.axhline(0, color='grey', lw=0.8, linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Confounding Gap (pp)\n(RCT gain − CONF gain)')
    ax.set_title('Confounding Robustness — Performance Drop from RCT → CONF (3-step)\n'
                 '(lower = more robust)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig8_confounding_robustness.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Figure: LSTM-DQN Subset Comparison ───────────────────────────────────────

SUBSET_ORDER = [
    ([0],       '{0}',     'Int. 0 only'),
    ([1],       '{1}',     'Int. 1 only'),
    ([2],       '{2}',     'Int. 2 only'),
    ([0, 1],    '{0,1}',   'Int. 0+1'),
    ([0, 2],    '{0,2}',   'Int. 0+2'),
    ([1, 2],    '{1,2}',   'Int. 1+2'),
    ([0, 1, 2], '{0,1,2}', 'All (joint)'),
]


def _subset_ids(active):
    return "".join(str(i) for i in sorted(active))


def fig_lstm_subsets(out_dir, subset_path, fallback_results, rct_subset_path=None):
    """LSTM-DQN gain vs Bank across 7 intervention subsets, CONF (+ RCT if present)."""
    if not os.path.exists(subset_path):
        print(f"[skip] fig_lstm_subsets: {subset_path} not found")
        return

    with open(subset_path) as f:
        subsets = json.load(f)

    if rct_subset_path and os.path.exists(rct_subset_path):
        with open(rct_subset_path) as f:
            subsets.update(json.load(f))

    # Bank reference per suffix: prefer the subset file's joint entry if it contains Bank,
    # else fall back to fallback_results[lstm_{suffix}_3].
    def bank_for(suffix):
        # Try subset file's lstm_{suffix}_joint
        joint = subsets.get(f"lstm_{suffix}_joint", {})
        if 'Bank' in joint:
            return joint['Bank'].get('mean'), joint['Bank'].get('std', 0)
        # Try standard all_results
        agg = (fallback_results or {}).get(f"lstm_{suffix}_{STEPS}", {}).get('aggregated', {})
        if 'Bank' in agg:
            return agg['Bank']['mean'], agg['Bank'].get('std', 0)
        return None, None

    # Collect data: gain (%) per subset per suffix
    suffixes_present = []
    data = {}   # suffix -> list of (mean_gain, std_gain)
    for suffix in ['RCT', 'CONF']:
        bank_m, bank_std = bank_for(suffix)
        if bank_m is None or bank_m <= 0:
            continue
        rows = []
        any_present = False
        for (active, _, _) in SUBSET_ORDER:
            ids = _subset_ids(active)
            entry = subsets.get(f"lstm_{suffix}_Int{ids}")
            if entry is None:
                # Fall back to joint for [0,1,2]
                if active == [0, 1, 2]:
                    joint = subsets.get(f"lstm_{suffix}_joint", {})
                    pol_key = next((k for k in joint if k not in ('Bank', 'Random')), None)
                    if pol_key:
                        pol_m   = joint[pol_key]['mean']
                        pol_std = joint[pol_key].get('std', 0)
                        gain = ((pol_m / bank_m) - 1) * 100
                        rel  = np.sqrt((pol_std/abs(pol_m))**2 + (bank_std/abs(bank_m))**2) if pol_m else 0
                        rows.append((gain, abs(gain) * rel))
                        any_present = True
                        continue
                rows.append((np.nan, 0))
                continue
            pol_m   = entry['mean']
            pol_std = entry.get('std', 0)
            gain = ((pol_m / bank_m) - 1) * 100
            rel  = np.sqrt((pol_std/abs(pol_m))**2 + (bank_std/abs(bank_m))**2) if pol_m else 0
            rows.append((gain, abs(gain) * rel))
            any_present = True
        if any_present:
            suffixes_present.append(suffix)
            data[suffix] = rows

    if not suffixes_present:
        print("[skip] fig_lstm_subsets: no usable subset entries")
        return

    # Plot grouped bars
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_subsets = len(SUBSET_ORDER)
    x = np.arange(n_subsets)
    n_suf = len(suffixes_present)
    width = 0.4 if n_suf > 1 else 0.6

    for si, suffix in enumerate(suffixes_present):
        gains = [g for g, _ in data[suffix]]
        errs  = [s for _, s in data[suffix]]
        offset = (si - (n_suf - 1) / 2) * width
        alpha = 0.9 if suffix == 'RCT' else 0.6
        hatch = '' if suffix == 'RCT' else '///'
        ax.bar(x + offset, gains, width * 0.9, yerr=errs, capsize=3,
               color=LSTM_BLUE, alpha=alpha, hatch=hatch,
               edgecolor='black', linewidth=0.5,
               label=suffix, error_kw={'lw': 1.2})

    ax.axhline(0, color='grey', lw=0.8, linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in SUBSET_ORDER], fontsize=10)
    ax.set_xlabel('Active intervention subset (LSTM-DQN controls these; bank policy otherwise)')
    ax.set_ylabel('% Gain vs Bank Policy')
    ax.set_title('LSTM-DQN — Gain by Intervention Subset (3-step task)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    legend_elems = []
    if 'RCT' in suffixes_present:
        legend_elems.append(mpatches.Patch(facecolor=LSTM_BLUE, alpha=0.9,
                                           edgecolor='black', label='RCT'))
    if 'CONF' in suffixes_present:
        legend_elems.append(mpatches.Patch(facecolor=LSTM_BLUE, alpha=0.6,
                                           edgecolor='black', hatch='///', label='CONF'))
    ax.legend(handles=legend_elems, frameon=False, fontsize=13,
              loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_lstm_subsets.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global METHODS
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results/all_results.json')
    parser.add_argument('--subsets', type=str, default='results/lstm_joint_vs_subset.json',
                        help='Path to LSTM-DQN CONF subset results JSON.')
    parser.add_argument('--subsets_rct', type=str, default='results/lstm_joint_vs_subset_rct.json',
                        help='Path to LSTM-DQN RCT subset results JSON (merged on top of --subsets).')
    parser.add_argument('--out',     type=str, default='results/figures')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Subset of methods to render (defaults to all known methods).')
    args = parser.parse_args()

    if args.methods:
        unknown = [m for m in args.methods if m not in METHOD_LABELS]
        if unknown:
            print(f"[ERROR] Unknown methods: {unknown}. Known: {list(METHOD_LABELS)}")
            sys.exit(1)
        METHODS = [m for m in METHODS if m in args.methods]

    results_path = os.path.join(script_dir, args.results)
    if not os.path.exists(results_path):
        print(f"[ERROR] Results file not found: {results_path}")
        sys.exit(1)

    results = load_results(results_path)
    out_dir = os.path.join(script_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    suffixes = []
    if any(f"_RCT_{STEPS}" in k for k in results):
        suffixes.append('RCT')
    if any(f"_CONF_{STEPS}" in k for k in results):
        suffixes.append('CONF')

    if not suffixes:
        print(f"[ERROR] No {STEPS}-step results found in JSON file.")
        sys.exit(1)

    print(f"\nGenerating figures from {len(results)} result entries")
    print(f"Suffixes available: {suffixes}  (3-step only)")
    print(f"Output directory: {out_dir}\n")

    fig2_absolute_performance(results, out_dir, suffixes)
    fig3_rct_vs_conf(results, out_dir)
    fig4_seed_variance(results, out_dir, suffixes)
    fig5_gain_heatmap(results, out_dir, suffixes)
    fig7_dqn_vs_procause(results, out_dir, suffixes)
    fig8_confounding_robustness(results, out_dir)
    fig_normalized_gap(results, out_dir)
    fig_lstm_subsets(out_dir, os.path.join(script_dir, args.subsets), results,
                     rct_subset_path=os.path.join(script_dir, args.subsets_rct))

    print(f"\n[OK] Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
