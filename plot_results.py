"""
Generate thesis figures from run_all_steps.py results.

Usage:
    python plot_results.py                           # read results/all_results.json
    python plot_results.py --results results/all_results.json
    python plot_results.py --out figures/            # output directory

Figures produced:
    fig1_marginal_contribution.pdf   — line chart: % gain vs Bank by steps, per method
    fig2_absolute_performance.pdf    — grouped bar chart: avg outcome by method × steps
    fig3_rct_vs_conf.pdf             — side-by-side RCT vs CONF for each method × steps
    fig4_seed_variance.pdf           — box plots of per-seed outcomes
    fig5_gain_heatmap.pdf            — heatmap: method × steps, colour = % gain vs Bank
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
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── Style ─────────────────────────────────────────────────────────────────────
METHODS   = ['kmeans', 'lstm', 'rims', 'multiModelCQL', 'singleModelCQL', 'procause_lstm', 'procause_econml']
METHOD_LABELS = {
    'kmeans': 'K-Means', 'lstm': 'LSTM-DQN', 'rims': 'RIMS',
    'multiModelCQL': 'CQL-MM', 'singleModelCQL': 'CQL-SM',
    'procause_lstm': 'ProCause-LSTM', 'procause_econml': 'ProCause-EconML',
}
COLORS    = {
    'kmeans': '#2196F3', 'lstm': '#FF9800', 'rims': '#4CAF50',
    'multiModelCQL': '#9C27B0', 'singleModelCQL': '#F44336',
    'procause_lstm': '#E91E63', 'procause_econml': '#009688',
}
STEP_LABELS = {1: '1-step\n(Int. 0)', 2: '2-step\n(Int. 0–1)', 3: '3-step\n(Int. 0–2)'}
SUFFIX_STYLES = {'RCT': '-', 'CONF': '--'}
SUFFIX_MARKERS = {'RCT': 'o', 'CONF': 's'}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(path):
    with open(path) as f:
        return json.load(f)


def get_agg(results, method, suffix, steps):
    key = f"{method}_{suffix}_{steps}"
    if key not in results:
        return None
    return results[key]['aggregated']


def get_gain(agg, bank_key='Bank'):
    """% gain of the method vs Bank."""
    if agg is None:
        return None, None
    bank_m = agg[bank_key]['mean']
    policy_key = [k for k in agg if k not in ('Bank', 'Random')][0]
    pol_m  = agg[policy_key]['mean']
    pol_std = agg[policy_key]['std']
    bank_std = agg[bank_key]['std']
    gain   = ((pol_m / bank_m) - 1) * 100 if bank_m > 0 else float('nan')
    # propagate std to % gain using delta method: σ_gain ≈ |pol/bank| * sqrt((σ_pol/pol)²+(σ_bank/bank)²) * 100
    if bank_m > 0 and pol_m != 0:
        rel_err = np.sqrt((pol_std / abs(pol_m))**2 + (bank_std / abs(bank_m))**2)
        gain_std = abs(gain) * rel_err
    else:
        gain_std = float('nan')
    return gain, gain_std


def get_per_seed_gains(results, method, suffix, steps):
    key = f"{method}_{suffix}_{steps}"
    if key not in results:
        return []
    agg = results[key]['aggregated']
    per_seed = results[key]['per_seed']
    policy_key = [k for k in agg if k not in ('Bank', 'Random')][0]
    gains = []
    for seed_data in per_seed.values():
        bank = seed_data.get('Bank', seed_data.get('Bank', None))
        pol  = seed_data.get(policy_key, None)
        if bank and pol and bank > 0:
            gains.append(((pol / bank) - 1) * 100)
    return gains


# ── Figure 1: Marginal Contribution Line Chart ────────────────────────────────

def fig1_marginal_contribution(results, out_dir, suffixes):
    n_suf = len(suffixes)
    fig, axes = plt.subplots(1, n_suf, figsize=(6 * n_suf, 5), sharey=True)
    if n_suf == 1:
        axes = [axes]

    for ax, suffix in zip(axes, suffixes):
        ax.axhline(0, color='grey', lw=0.8, linestyle='--', alpha=0.6, label='Bank baseline (0%)')

        for method in METHODS:
            gains, stds, xs = [], [], []
            for steps in [1, 2, 3]:
                g, s = get_gain(get_agg(results, method, suffix, steps))
                if g is not None:
                    gains.append(g)
                    stds.append(s if not np.isnan(s) else 0)
                    xs.append(steps)

            if gains:
                ax.plot(xs, gains, color=COLORS[method], marker='o', linewidth=2,
                        markersize=7, label=METHOD_LABELS[method])
                ax.fill_between(xs,
                                [g - s for g, s in zip(gains, stds)],
                                [g + s for g, s in zip(gains, stds)],
                                color=COLORS[method], alpha=0.15)

        ax.set_title(f'{suffix} Data')
        ax.set_xlabel('Number of Controlled Interventions')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['1-step\n(Int. 0 only)', '2-step\n(Int. 0–1)', '3-step\n(All)'])
        ax.set_ylabel('% Gain vs Bank Policy' if suffix == suffixes[0] else '')
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Marginal Contribution of Each Intervention', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_marginal_contribution.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Figure 2: Absolute Performance Grouped Bar Chart ─────────────────────────

def fig2_absolute_performance(results, out_dir, suffixes):
    n_suf = len(suffixes)
    fig, axes = plt.subplots(1, n_suf, figsize=(7 * n_suf, 6), sharey=True)
    if n_suf == 1:
        axes = [axes]

    available_methods = [m for m in METHODS if any(
        get_agg(results, m, s, st) for s in suffixes for st in [1, 2, 3])]
    n_methods = len(available_methods)
    width = 0.8 / n_methods

    for ax, suffix in zip(axes, suffixes):
        steps_list = [1, 2, 3]
        n_steps = len(steps_list)
        x = np.arange(n_steps) * (n_methods * width + 0.6)

        bank_vals = []
        for steps in steps_list:
            agg = get_agg(results, available_methods[0], suffix, steps)
            if agg:
                bank_vals.append(agg['Bank']['mean'])
        if bank_vals:
            ax.axhline(np.mean(bank_vals), color='black', lw=1.5, linestyle='--',
                       label='Bank policy', zorder=5)

        for mi, method in enumerate(available_methods):
            means, errs = [], []
            for steps in steps_list:
                agg = get_agg(results, method, suffix, steps)
                if agg is None:
                    means.append(0); errs.append(0); continue
                policy_key = [k for k in agg if k not in ('Bank', 'Random')][0]
                means.append(agg[policy_key]['mean'])
                errs.append(agg[policy_key]['std'])

            offset = (mi - n_methods / 2 + 0.5) * width
            ax.bar(x + offset, means, width * 0.9, yerr=errs, capsize=3,
                   color=COLORS[method], label=METHOD_LABELS[method],
                   alpha=0.85, error_kw={'lw': 1})

        ax.set_title(f'{suffix} Data')
        ax.set_xlabel('Steps')
        ax.set_xticks(x)
        ax.set_xticklabels(['1-step\n(Int. 0)', '2-step\n(Int. 0–1)', '3-step\n(All)'])
        ax.set_ylabel('Average Outcome' if suffix == suffixes[0] else '')
        ax.legend(frameon=False, ncol=2, loc='upper left', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Absolute Performance by Method and Step Count', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_absolute_performance.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Figure 3: RCT vs CONF Comparison ─────────────────────────────────────────

def fig3_rct_vs_conf(results, out_dir):
    available = [m for m in METHODS if f"{m}_CONF_3" in results or f"{m}_CONF_1" in results]
    if not available:
        print("[skip] fig3: CONF results not available")
        return

    n_methods = len(available)
    ncols = min(4, n_methods)
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), sharey=True)
    axes = np.atleast_2d(axes)
    flat_axes = axes.flatten()

    for idx, method in enumerate(available):
        ax = flat_axes[idx]
        steps_list = [1, 2, 3]
        x = np.arange(len(steps_list))
        width = 0.32

        for si, suffix in enumerate(['RCT', 'CONF']):
            gains, errs = [], []
            for steps in steps_list:
                g, s = get_gain(get_agg(results, method, suffix, steps))
                gains.append(g if g is not None else 0)
                errs.append(s if (s is not None and not np.isnan(s)) else 0)

            offset = (si - 0.5) * width
            alpha = 0.9 if suffix == 'RCT' else 0.45
            hatch = '' if suffix == 'RCT' else '///'
            ax.bar(x + offset, gains, width, yerr=errs, capsize=4,
                   color=COLORS[method], alpha=alpha, hatch=hatch, label=suffix,
                   error_kw={'lw': 1.2})

        ax.axhline(0, color='grey', lw=0.8, linestyle='--', alpha=0.6)
        ax.set_title(METHOD_LABELS[method], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['1-step', '2-step', '3-step'])
        if idx % ncols == 0:
            ax.set_ylabel('% Gain vs Bank Policy')
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    for idx in range(n_methods, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle('RCT vs. Confounded Data: % Gain over Bank Policy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_rct_vs_conf.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Figure 4: Seed Variance Box Plots ────────────────────────────────────────

def fig4_seed_variance(results, out_dir, suffixes):
    n_suf = len(suffixes)
    fig, axes = plt.subplots(1, n_suf, figsize=(8 * n_suf, 5), sharey=True)
    if n_suf == 1:
        axes = [axes]

    for ax, suffix in zip(axes, suffixes):
        data_all, labels_all, colors_all, positions = [], [], [], []
        pos = 1
        tick_pos, tick_labels = [], []
        step_groups = []

        for steps in [1, 2, 3]:
            group_start = pos
            for method in METHODS:
                gains = get_per_seed_gains(results, method, suffix, steps)
                if gains:
                    data_all.append(gains)
                    labels_all.append(METHOD_LABELS[method])
                    colors_all.append(COLORS[method])
                    positions.append(pos)
                pos += 1
            # gap between step groups
            step_groups.append((group_start, pos - 1, steps))
            pos += 0.5

        bp = ax.boxplot(data_all, positions=positions, widths=0.6,
                        patch_artist=True, notch=False,
                        medianprops={'color': 'black', 'lw': 2},
                        whiskerprops={'lw': 1.2}, capprops={'lw': 1.2},
                        flierprops={'marker': 'x', 'markersize': 5})

        for patch, color in zip(bp['boxes'], colors_all):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Step group labels
        for (start, end, steps) in step_groups:
            mid = (start + end) / 2
            ax.text(mid, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -5,
                    f'{steps}-step', ha='center', va='top',
                    fontsize=9, color='grey')
            if end < positions[-1]:
                ax.axvline(end + 0.25, color='lightgrey', lw=0.8)

        ax.axhline(0, color='grey', lw=0.8, linestyle='--', alpha=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_all, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{suffix} Data')
        ax.set_ylabel('% Gain vs Bank Policy' if suffix == suffixes[0] else '')
        ax.grid(axis='y', alpha=0.3)

        legend_patches = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m]) for m in METHODS]
        ax.legend(handles=legend_patches, frameon=False)

    fig.suptitle('Per-Seed Variance in % Gain over Bank Policy', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_seed_variance.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Figure 5: Gain Heatmap ────────────────────────────────────────────────────

def fig5_gain_heatmap(results, out_dir, suffixes):
    n_suf = len(suffixes)
    fig, axes = plt.subplots(1, n_suf, figsize=(5 * n_suf, 4.5))
    if n_suf == 1:
        axes = [axes]

    for ax, suffix in zip(axes, suffixes):
        # rows = methods, cols = steps
        data = np.full((len(METHODS), 3), np.nan)
        for mi, method in enumerate(METHODS):
            for si, steps in enumerate([1, 2, 3]):
                g, _ = get_gain(get_agg(results, method, suffix, steps))
                if g is not None:
                    data[mi, si] = g

        # Diverging colormap centred at 0
        vmax = np.nanmax(np.abs(data))
        im = ax.imshow(data, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['1-step', '2-step', '3-step'])
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS])
        ax.set_title(f'{suffix}')

        for mi in range(len(METHODS)):
            for si in range(3):
                val = data[mi, si]
                text = f'{val:+.1f}%' if not np.isnan(val) else 'N/A'
                ax.text(si, mi, text, ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if abs(val) > vmax * 0.5 else 'black')

        plt.colorbar(im, ax=ax, label='% Gain vs Bank', shrink=0.8)

    fig.suptitle('% Gain over Bank Policy — Method × Steps', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig5_gain_heatmap.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Figure 6: Incremental Gain (Δ per additional step) ───────────────────────

def fig6_incremental_gain(results, out_dir, suffixes):
    """Bar chart showing marginal gain from adding each intervention."""
    n_suf = len(suffixes)
    fig, axes = plt.subplots(1, n_suf, figsize=(7 * n_suf, 6), sharey=True)
    if n_suf == 1:
        axes = [axes]

    available_methods = []
    for method in METHODS:
        for suffix in suffixes:
            g1, _ = get_gain(get_agg(results, method, suffix, 1))
            g2, _ = get_gain(get_agg(results, method, suffix, 2))
            g3, _ = get_gain(get_agg(results, method, suffix, 3))
            if all(v is not None for v in [g1, g2, g3]):
                if method not in available_methods:
                    available_methods.append(method)
                break

    n_methods = len(available_methods)
    width = 0.8 / n_methods

    for ax, suffix in zip(axes, suffixes):
        x = np.arange(2) * (n_methods * width + 0.6)

        for mi, method in enumerate(available_methods):
            g1, _ = get_gain(get_agg(results, method, suffix, 1))
            g2, _ = get_gain(get_agg(results, method, suffix, 2))
            g3, _ = get_gain(get_agg(results, method, suffix, 3))

            if any(v is None for v in [g1, g2, g3]):
                continue

            deltas = [g2 - g1, g3 - g2]
            offset = (mi - n_methods / 2 + 0.5) * width
            ax.bar(x + offset, deltas, width * 0.9, color=COLORS[method],
                   label=METHOD_LABELS[method], alpha=0.85)

        ax.axhline(0, color='grey', lw=0.8, linestyle='--', alpha=0.6)
        ax.set_title(f'{suffix} Data')
        ax.set_xlabel('Intervention Added')
        ax.set_xticks(x)
        ax.set_xticklabels(['Adding Int. 1\n(1-step → 2-step)', 'Adding Int. 2\n(2-step → 3-step)'])
        ax.set_ylabel('Incremental % Gain vs Bank' if suffix == suffixes[0] else '')
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Incremental Gain per Additional Controlled Intervention', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig6_incremental_gain.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results/all_results.json')
    parser.add_argument('--out',     type=str, default='results/figures')
    args = parser.parse_args()

    results_path = os.path.join(script_dir, args.results)
    if not os.path.exists(results_path):
        print(f"[ERROR] Results file not found: {results_path}")
        print("        Run `python run_all_steps.py` first to generate results.")
        sys.exit(1)

    results = load_results(results_path)
    out_dir = os.path.join(script_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Determine which suffixes are available
    suffixes = []
    if any('_RCT_' in k for k in results):
        suffixes.append('RCT')
    if any('_CONF_' in k for k in results):
        suffixes.append('CONF')

    if not suffixes:
        print("[ERROR] No results found in JSON file.")
        sys.exit(1)

    print(f"\nGenerating figures from {len(results)} result entries")
    print(f"Suffixes available: {suffixes}")
    print(f"Output directory: {out_dir}\n")

    fig1_marginal_contribution(results, out_dir, suffixes)
    fig2_absolute_performance(results, out_dir, suffixes)
    fig3_rct_vs_conf(results, out_dir)
    fig4_seed_variance(results, out_dir, suffixes)
    fig5_gain_heatmap(results, out_dir, suffixes)
    fig6_incremental_gain(results, out_dir, suffixes)

    print(f"\n[OK] All figures saved to {out_dir}/")
    print("     Files: fig1_*.pdf/.png through fig6_*.pdf/.png")


if __name__ == "__main__":
    main()
