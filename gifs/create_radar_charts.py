import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# RADAR CHART GENERATOR — IID, NONIID Objective + Key Metrics
# ============================================================================

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

# Colors
COLORS = {
    'higat': '#0D47A1',
    'mappo': '#2E7D32',
    'ni_f2': '#E65100',
    'ni_f2f3': '#C62828',
    'ni_f1f3': '#6A1B9A',
    'ni_all': '#00838F',
    'ni_f3': '#AD1457',
    'ni_f1f2': '#F57F17',
    'gat_noedge': '#558B2F',
    'flatgat': '#4527A0',
    'ni_f1': '#BF360C',
    'mat': '#37474F',
    'happo': '#880E4F',
}

# ============================================================================
# 1. IID OBJECTIVE COMPONENTS RADAR
# ============================================================================
print("Generating IID Objective Radar...")

iid_labels = [
    'TSR\n(×0.30)',
    'Fairness\n(×0.20)',
    'Energy Eff\n(×0.15)',
    'Step Reward\n(×0.12)',
    'Throughput\n(×0.10)',
    'Fairness CV\n(×0.07)',
    'Entropy\n(×0.04)',
    'Stability\n(×0.02)'
]

# Values normalized per-component (extracted from component breakdown logic)
# These are the RAW component values before weighting, normalized to [0,1]
iid_data = {
    'HiGAT-HAPPO (Full)': [0.97, 0.99, 0.70, 0.72, 0.82, 0.72, 0.55, 0.79],
    'MAPPO (No FL)':      [0.96, 0.98, 0.45, 0.79, 0.81, 0.58, 0.80, 0.49],
    'MAT (No FL)':        [1.00, 0.96, 0.05, 0.03, 0.14, 0.72, 0.00, 0.00],
    'GAT-NoEdge-HAPPO':   [0.95, 0.99, 0.67, 0.67, 0.80, 0.61, 0.54, 0.79],
    'FlatGAT-HAPPO':      [0.95, 0.99, 0.60, 0.61, 0.76, 0.66, 0.54, 0.89],
    'HAPPO (No FL)':      [1.00, 0.88, 0.01, 0.05, 0.08, 0.12, 0.00, 0.00],
}

num_vars = len(iid_labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the circle

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for name, values in iid_data.items():
    values_closed = values + values[:1]
    ax.plot(angles, values_closed, color=COLORS.get(name.lower().replace(' ', '_')[:6], '#333333'), 
            linewidth=2, label=name, alpha=0.85)
    ax.fill(angles, values_closed, color=COLORS.get(name.lower().replace(' ', '_')[:6], '#333333'), 
            alpha=0.08)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(iid_labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
ax.spines['polar'].set_color('#CCCCCC')
ax.grid(color='#E0E0E0', linewidth=0.8)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.savefig('/home/ahmed/Desktop/radar_iid_objective.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved radar_iid_objective.png")

# ============================================================================
# 2. NONIID OBJECTIVE COMPONENTS RADAR
# ============================================================================
print("Generating NONIID Objective Radar...")

noniid_data = {
    'NI: F2 Only (Energy)':  [0.98, 1.00, 0.85, 0.73, 0.87, 0.82, 0.97, 1.00],
    'NI: F2+F3 (Ene+Fair)':  [0.97, 0.99, 0.89, 0.74, 0.91, 0.75, 0.97, 0.46],
    'NI: F1+F3 (Rew+Fair)':  [0.96, 0.99, 0.95, 0.78, 0.93, 0.65, 0.97, 0.60],
    'MAPPO (No FL)':         [0.95, 0.94, 0.55, 0.81, 0.97, 0.35, 0.72, 0.31],
    'NI: F1+F2+F3 (All)':    [0.93, 0.98, 0.91, 0.74, 0.91, 0.55, 0.97, 0.97],
    'HiGAT-HAPPO (Full)':    [0.91, 0.97, 0.95, 0.75, 0.92, 0.46, 0.96, 1.00],
    'NI: F3 Only (Fairness)': [0.92, 0.97, 0.87, 0.71, 0.89, 0.48, 0.97, 0.83],
    'NI: F1+F2 (Rew+Ene)':   [0.91, 0.96, 0.89, 0.72, 0.90, 0.42, 0.97, 1.00],
    'GAT-NoEdge-HAPPO':      [0.91, 0.96, 0.95, 0.74, 0.92, 0.41, 0.96, 1.00],
    'FlatGAT-HAPPO':         [0.91, 0.95, 0.87, 0.71, 0.92, 0.39, 0.97, 0.96],
    'NI: F1 Only (Reward)':  [0.91, 0.92, 0.80, 0.68, 0.84, 0.33, 0.97, 0.62],
    'MAT (No FL)':           [1.00, 0.78, 0.01, 0.00, 0.10, 0.01, 0.00, 0.00],
    'HAPPO (No FL)':         [1.00, 0.77, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00],
}

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

for name, values in noniid_data.items():
    values_closed = values + values[:1]
    ax.plot(angles, values_closed, color=COLORS.get(name.lower().replace(' ', '_')[:6], '#333333'), 
            linewidth=2, label=name, alpha=0.85)
    ax.fill(angles, values_closed, color=COLORS.get(name.lower().replace(' ', '_')[:6], '#333333'), 
            alpha=0.06)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(iid_labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
ax.spines['polar'].set_color('#CCCCCC')
ax.grid(color='#E0E0E0', linewidth=0.8)
ax.legend(loc='upper right', bbox_to_anchor=(1.40, 1.05), fontsize=7.5, framealpha=0.9, ncol=2)
plt.tight_layout()
plt.savefig('/home/ahmed/Desktop/radar_noniid_objective.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved radar_noniid_objective.png")

# ============================================================================
# 3. KEY METRICS RADAR — IID Top Performers
# ============================================================================
print("Generating IID Top Metrics Radar...")

iid_metric_labels = [
    'TSR\n(%)',
    'Jain\nFairness',
    'Energy/Task\n(lower=better)',
    'Step\nReward',
    'Throughput',
    'Fairness\nCV (lower)',
    'Entropy',
    'TSR\nStability'
]

# Raw values normalized to [0,1] via min-max per metric
# TSR: MAT=100 -> 1.0, HAPPO=100 -> 1.0, FlatGAT=95.09 -> 0.95
iid_top_data = {
    'HiGAT-HAPPO':  [0.97, 0.99, 0.82, 0.72, 0.82, 0.72, 0.55, 0.79],
    'MAPPO':        [0.96, 0.98, 0.45, 0.79, 0.81, 0.58, 0.80, 0.49],
    'MAT':          [1.00, 0.96, 0.05, 0.03, 0.14, 0.72, 0.00, 0.00],
}

num_vars2 = len(iid_metric_labels)
angles2 = np.linspace(0, 2 * np.pi, num_vars2, endpoint=False).tolist()
angles2 += angles2[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for name, values in iid_top_data.items():
    values_closed = values + values[:1]
    ax.plot(angles2, values_closed, color=COLORS.get(name.lower(), '#333333'), 
            linewidth=2.5, label=name, alpha=0.9)
    ax.fill(angles2, values_closed, color=COLORS.get(name.lower(), '#333333'), alpha=0.10)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles2[:-1])
ax.set_xticklabels(iid_metric_labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
ax.spines['polar'].set_color('#CCCCCC')
ax.grid(color='#E0E0E0', linewidth=0.8)
ax.legend(loc='upper right', bbox_to_anchor=(1.30, 1.10), fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig('/home/ahmed/Desktop/radar_iid_top3.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved radar_iid_top3.png")

# ============================================================================
# 4. KEY METRICS RADAR — NONIID Top Performers
# ============================================================================
print("Generating NONIID Top Metrics Radar...")

noniid_top_data = {
    'NI: F2 (Energy)':  [0.98, 1.00, 0.85, 0.73, 0.87, 0.82, 0.97, 1.00],
    'NI: F2+F3':        [0.97, 0.99, 0.89, 0.74, 0.91, 0.75, 0.97, 0.46],
    'NI: F1+F3':        [0.96, 0.99, 0.95, 0.78, 0.93, 0.65, 0.97, 0.60],
    'MAPPO':            [0.95, 0.94, 0.55, 0.81, 0.97, 0.35, 0.72, 0.31],
    'HiGAT (Full)':     [0.91, 0.97, 0.95, 0.75, 0.92, 0.46, 0.96, 1.00],
}

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for name, values in noniid_top_data.items():
    values_closed = values + values[:1]
    ax.plot(angles2, values_closed, color=COLORS.get(name.lower().replace(' ', '_')[:6], '#333333'), 
            linewidth=2.5, label=name, alpha=0.9)
    ax.fill(angles2, values_closed, color=COLORS.get(name.lower().replace(' ', '_')[:6], '#333333'), alpha=0.08)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles2[:-1])
ax.set_xticklabels(iid_metric_labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
ax.spines['polar'].set_color('#CCCCCC')
ax.grid(color='#E0E0E0', linewidth=0.8)
ax.legend(loc='upper right', bbox_to_anchor=(1.30, 1.10), fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig('/home/ahmed/Desktop/radar_noniid_top5.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved radar_noniid_top5.png")

# ============================================================================
# 5. ISAAC OBJECTIVE RADAR
# ============================================================================
print("Generating Isaac Objective Radar...")

isaac_labels = [
    'TSR\n(×0.30)',
    'Fairness\n(×0.20)',
    'Energy Eff\n(×0.15)',
    'Step Reward\n(×0.12)',
    'Throughput\n(×0.10)',
    'Fairness Stab\n(×0.07)',
    'Comp Eff\n(×0.04)',
    'Seed Stab\n(×0.02)'
]

isaac_data = {
    'MAPPO':          [0.95, 0.63, 0.85, 0.78, 0.60, 0.70, 0.00, 0.98],
    'HiGAT NI-F1F2F3': [0.77, 0.60, 0.65, 0.65, 0.46, 0.00, 0.50, 0.94],
    'HiGAT no_fl':    [0.59, 0.62, 0.45, 0.45, 0.35, 0.50, 1.00, 0.83],
}

num_vars3 = len(isaac_labels)
angles3 = np.linspace(0, 2 * np.pi, num_vars3, endpoint=False).tolist()
angles3 += angles3[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

isaac_colors = ['#0D47A1', '#2E7D32', '#E65100']
for i, (name, values) in enumerate(isaac_data.items()):
    values_closed = values + values[:1]
    ax.plot(angles3, values_closed, color=isaac_colors[i], linewidth=2.5, label=name, alpha=0.9)
    ax.fill(angles3, values_closed, color=isaac_colors[i], alpha=0.10)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles3[:-1])
ax.set_xticklabels(isaac_labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
ax.spines['polar'].set_color('#CCCCCC')
ax.grid(color='#E0E0E0', linewidth=0.8)
ax.legend(loc='upper right', bbox_to_anchor=(1.30, 1.10), fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig('/home/ahmed/Desktop/radar_isaac_objective.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved radar_isaac_objective.png")

print("\n=== ALL RADAR CHARTS GENERATED ===")
print("Files saved to /home/ahmed/Desktop/:")
print("  radar_iid_objective.png")
print("  radar_noniid_objective.png")
print("  radar_iid_top3.png")
print("  radar_noniid_top5.png")
print("  radar_isaac_objective.png")
