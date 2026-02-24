"""
Generate Performance Comparison Graphs
=======================================
Creates visualizations comparing all methods across 1, 2, and 3 intervention steps.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Create output directory
output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

# Data: Gain vs Bank (%) for each method at 1, 2, 3 steps
# Format: method_name: [1-step, 2-step, 3-step]

# RCT Data
rct_data = {
    'RIMS': [24.18, 189.17, 415.09],
    'KMeans': [26.06, 197.52, 409.84],
    'LSTM': [23.25, 179.18, 406.16],
    'CQL Unified': [25.75, 33.65, 343.54],
    'CQL multiModel': [25.08, 24.53, 290.50],
}

# CONFOUNDED Data
confounded_data = {
    'RIMS': [-119.54, 132.12, 276.77],
    'KMeans': [-1.74, 90.89, 152.54],
    'LSTM': [-119.24, -5.96, 235.17],
    'CQL Unified': [1.67, 0.99, 262.23],
    'CQL multiModel': [-6.52, 1.93, 293.30],
}

# X-axis: number of steps
steps = [1, 2, 3]

# Color scheme
colors = {
    'RIMS': '#e74c3c',          # Red
    'KMeans': '#3498db',         # Blue
    'LSTM': '#2ecc71',           # Green
    'CQL Unified': '#f39c12',    # Orange
    'CQL multiModel': '#9b59b6', # Purple
}

markers = {
    'RIMS': 'o',
    'KMeans': 's',
    'LSTM': '^',
    'CQL Unified': 'D',
    'CQL multiModel': 'v',
}

# ============================================================================
# GRAPH 1: RCT Performance Comparison
# ============================================================================
print("Generating Graph 1: RCT Performance Comparison...")

fig, ax = plt.subplots(figsize=(14, 9))

for method, gains in rct_data.items():
    ax.plot(steps, gains,
            marker=markers[method],
            color=colors[method],
            linewidth=2.5,
            markersize=10,
            label=method,
            alpha=0.8)

ax.set_xlabel('Number of Intervention Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('Gain vs Bank Baseline (%)', fontsize=14, fontweight='bold')
ax.set_title('Performance Comparison: RCT Data\n(Higher is Better)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(steps)
ax.set_xticklabels(['Single-Step\n(1 Intervention)',
                    'Multi-Step\n(2 Interventions)',
                    'Triple-Step\n(3 Interventions)'])
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# Add value labels on points
for method, gains in rct_data.items():
    for i, (x, y) in enumerate(zip(steps, gains)):
        if i == 2:  # Only label the final point to avoid clutter
            ax.annotate(f'{y:.1f}%',
                       xy=(x, y),
                       xytext=(10, -5),
                       textcoords='offset points',
                       fontsize=10,
                       color=colors[method],
                       fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rct_performance_comparison.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/rct_performance_comparison.png")
plt.close()

# ============================================================================
# GRAPH 2: CONFOUNDED Performance Comparison
# ============================================================================
print("Generating Graph 2: CONFOUNDED Performance Comparison...")

fig, ax = plt.subplots(figsize=(14, 9))

for method, gains in confounded_data.items():
    ax.plot(steps, gains,
            marker=markers[method],
            color=colors[method],
            linewidth=2.5,
            markersize=10,
            label=method,
            alpha=0.8)

ax.set_xlabel('Number of Intervention Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('Gain vs Bank Baseline (%)', fontsize=14, fontweight='bold')
ax.set_title('Performance Comparison: CONFOUNDED Data\n(Higher is Better)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(steps)
ax.set_xticklabels(['Single-Step\n(1 Intervention)',
                    'Multi-Step\n(2 Interventions)',
                    'Triple-Step\n(3 Interventions)'])
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Bank Baseline')

# Add value labels on final points
for method, gains in confounded_data.items():
    x, y = steps[-1], gains[-1]
    ax.annotate(f'{y:.1f}%',
               xy=(x, y),
               xytext=(10, -5),
               textcoords='offset points',
               fontsize=10,
               color=colors[method],
               fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confounded_performance_comparison.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/confounded_performance_comparison.png")
plt.close()

# ============================================================================
# GRAPH 3: Combined RCT vs CONFOUNDED (Top 3 Methods)
# ============================================================================
print("Generating Graph 3: RCT vs CONFOUNDED Comparison (Top Methods)...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

top_methods = ['RIMS', 'KMeans', 'LSTM']

# RCT subplot
for method in top_methods:
    ax1.plot(steps, rct_data[method],
            marker=markers[method],
            color=colors[method],
            linewidth=2.5,
            markersize=10,
            label=method,
            alpha=0.8)

ax1.set_xlabel('Number of Intervention Steps', fontsize=13, fontweight='bold')
ax1.set_ylabel('Gain vs Bank (%)', fontsize=13, fontweight='bold')
ax1.set_title('RCT Data', fontsize=15, fontweight='bold', pad=15)
ax1.set_xticks(steps)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, framealpha=0.9)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# CONFOUNDED subplot
for method in top_methods:
    ax2.plot(steps, confounded_data[method],
            marker=markers[method],
            color=colors[method],
            linewidth=2.5,
            markersize=10,
            label=method,
            alpha=0.8)

ax2.set_xlabel('Number of Intervention Steps', fontsize=13, fontweight='bold')
ax2.set_ylabel('Gain vs Bank (%)', fontsize=13, fontweight='bold')
ax2.set_title('CONFOUNDED Data', fontsize=15, fontweight='bold', pad=15)
ax2.set_xticks(steps)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11, framealpha=0.9)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

plt.suptitle('Performance Comparison: Top 3 Methods (RCT vs CONFOUNDED)',
             fontsize=17, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rct_vs_confounded_top3.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/rct_vs_confounded_top3.png")
plt.close()

# ============================================================================
# GRAPH 4: Bar Chart - 3-Step Performance Comparison
# ============================================================================
print("Generating Graph 4: 3-Step Performance Bar Chart...")

fig, ax = plt.subplots(figsize=(14, 8))

methods = list(rct_data.keys())
rct_3step = [rct_data[m][2] for m in methods]
conf_3step = [confounded_data[m][2] for m in methods]

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, rct_3step, width, label='RCT', alpha=0.8, color='#3498db')
bars2 = ax.bar(x + width/2, conf_3step, width, label='CONFOUNDED', alpha=0.8, color='#e74c3c')

ax.set_xlabel('Method', fontsize=14, fontweight='bold')
ax.set_ylabel('Gain vs Bank (%)', fontsize=14, fontweight='bold')
ax.set_title('Triple-Step (3 Interventions) Performance Comparison\nRCT vs CONFOUNDED Data',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3step_bar_comparison.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/3step_bar_comparison.png")
plt.close()

# ============================================================================
# GRAPH 5: Scaling Factor (Performance Growth)
# ============================================================================
print("Generating Graph 5: Performance Scaling Analysis...")

fig, ax = plt.subplots(figsize=(14, 9))

# Calculate scaling: how much does performance improve from 1-step to 3-step
for method in rct_data.keys():
    # Normalize to 1-step performance (set 1-step = 100%)
    baseline = rct_data[method][0]
    if baseline > 0:
        normalized = [(gain / baseline) * 100 for gain in rct_data[method]]
    else:
        normalized = [100, 100, 100]  # If baseline is 0 or negative

    ax.plot(steps, normalized,
            marker=markers[method],
            color=colors[method],
            linewidth=2.5,
            markersize=10,
            label=method,
            alpha=0.8)

ax.set_xlabel('Number of Intervention Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('Performance Scaling (1-Step = 100%)', fontsize=14, fontweight='bold')
ax.set_title('Performance Scaling: How Methods Improve with More Interventions (RCT)\n(Higher = Better Scaling)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(steps)
ax.set_xticklabels(['1 Step\n(Baseline)', '2 Steps', '3 Steps'])
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax.axhline(y=100, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='1-Step Baseline')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_scaling_analysis.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/performance_scaling_analysis.png")
plt.close()

# ============================================================================
# GRAPH 6: Robustness Analysis (RCT - CONFOUNDED difference)
# ============================================================================
print("Generating Graph 6: Robustness Analysis...")

fig, ax = plt.subplots(figsize=(14, 9))

# Calculate RCT advantage (RCT gain - CONFOUNDED gain)
for method in rct_data.keys():
    rct_advantage = [rct_data[method][i] - confounded_data[method][i] for i in range(3)]

    ax.plot(steps, rct_advantage,
            marker=markers[method],
            color=colors[method],
            linewidth=2.5,
            markersize=10,
            label=method,
            alpha=0.8)

ax.set_xlabel('Number of Intervention Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('RCT Advantage (RCT % - CONFOUNDED %)', fontsize=14, fontweight='bold')
ax.set_title('Robustness Analysis: RCT Advantage Over CONFOUNDED Data\n(Lower = More Robust to Confounding)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(steps)
ax.set_xticklabels(['1 Step', '2 Steps', '3 Steps'])
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7,
          label='Perfect Robustness (CONF = RCT)')

# Highlight CQL multiModel as the most robust
for i, (x, y) in enumerate(zip(steps, [rct_data['CQL multiModel'][i] - confounded_data['CQL multiModel'][i] for i in range(3)])):
    if i == 2:
        ax.annotate('Most Robust',
                   xy=(x, y),
                   xytext=(20, 20),
                   textcoords='offset points',
                   fontsize=11,
                   color=colors['CQL multiModel'],
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', color=colors['CQL multiModel'], lw=2))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'robustness_analysis.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/robustness_analysis.png")
plt.close()

print("\n" + "="*70)
print("ALL GRAPHS GENERATED SUCCESSFULLY!")
print("="*70)
print(f"\nGraphs saved in: {output_dir}/")
print("\nGenerated graphs:")
print("  1. rct_performance_comparison.png - RCT performance across 1/2/3 steps")
print("  2. confounded_performance_comparison.png - CONFOUNDED performance across 1/2/3 steps")
print("  3. rct_vs_confounded_top3.png - Side-by-side comparison of top 3 methods")
print("  4. 3step_bar_comparison.png - Bar chart comparing 3-step performance")
print("  5. performance_scaling_analysis.png - How methods scale with more interventions")
print("  6. robustness_analysis.png - RCT advantage analysis (robustness to confounding)")
