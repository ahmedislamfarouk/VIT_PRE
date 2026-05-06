import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# RADAR CHART GENERATOR — Enhanced Version for "Very Small Diffs"
# ============================================================================

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Colors - Modern & Distinct
COLORS = {
    'higat': '#1A237E',  # Indigo
    'mappo': '#2E7D32',  # Green
    'ni_f2': '#E65100',  # Orange
    'ni_f2f3': '#C62828', # Red
    'ni_f1f3': '#6A1B9A', # Purple
    'ni_all': '#00838F',  # Cyan
    'ni_f3': '#AD1457',  # Pink
    'ni_f1f2': '#F57F17', # Yellow-Orange
    'gat_noedge': '#558B2F', # Light Green
    'flatgat': '#4527A0', # Deep Purple
    'ni_f1': '#BF360C',  # Deep Orange
    'mat': '#37474F',    # Blue Grey
    'happo': '#880E4F',  # Maroon
}

# ============================================================================
# 1. ZOOMED NONIID TOP PERFORMERS (The "Very Small Diffs")
# ============================================================================
print("Generating Zoomed NONIID Top Performers Radar...")

labels = ['TSR', 'Fairness', 'EnergyEff', 'StepRew', 'Throughput', 'FairStab', 'Entropy', 'Stability']

# Data for top 3 NONIID algorithms — showing the extremely close results
# Values from the NONIID Ranking table
noniid_top_data = {
    'NI: F2 Only (Energy)':  [0.98, 1.00, 0.85, 0.73, 0.87, 0.82, 0.97, 1.00],
    'NI: F2+F3 (Ene+Fair)':  [0.97, 0.99, 0.89, 0.74, 0.91, 0.75, 0.97, 0.46],
    'NI: F1+F3 (Rew+Fair)':  [0.96, 0.99, 0.95, 0.78, 0.93, 0.65, 0.97, 0.60],
}

num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for name, values in noniid_top_data.items():
    values_closed = values + values[:1]
    color = COLORS.get(name.lower().replace(' ', '_').replace(':', '')[:7], '#333333')
    ax.plot(angles, values_closed, color=color, linewidth=3, label=name, alpha=0.9, marker='o', markersize=4)
    ax.fill(angles, values_closed, color=color, alpha=0.05)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax.set_ylim(0.4, 1.05) 
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=9, color='gray')
ax.spines['polar'].set_color('#CCCCCC')
ax.grid(color='#E0E0E0', linewidth=1.0)
ax.set_title("NONIID Top-3 Performance (Zoomed 0.4-1.0)", va='bottom', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=10, frameon=True, shadow=True, ncol=1)

plt.tight_layout()
plt.savefig('/home/ahmed/Desktop/radar_noniid_zoomed.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# 2. ISAAC ROBUSTNESS RADAR (Seed Performance)
# ============================================================================
print("Generating Isaac Robustness Radar...")

isaac_seed_labels = ['Seed 42 TSR', 'Seed 123 TSR', 'Seed 456 TSR', 'Seed 42 Jain', 'Seed 123 Jain', 'Seed 456 Jain']

# Normalized metrics from per-seed table
# TSR: Max=97.7, Jain: Max=0.679
isaac_robustness = {
    'MAPPO': [97.7/97.7, 92.4/97.7, 94.2/97.7, 0.672/0.679, 0.610/0.679, 0.617/0.679],
    'HiGAT NI-F1F2F3': [82.0/97.7, 77.1/97.7, 71.1/97.7, 0.675/0.679, 0.636/0.679, 0.489/0.679],
    'HiGAT no_fl': [69.6/97.7, 63.5/97.7, 45.3/97.7, 0.679/0.679, 0.549/0.679, 0.636/0.679],
}

num_vars_r = len(isaac_seed_labels)
angles_r = np.linspace(0, 2 * np.pi, num_vars_r, endpoint=False).tolist()
angles_r += angles_r[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for name, values in isaac_robustness.items():
    values_closed = values + values[:1]
    color = COLORS.get(name.lower().split()[0], '#333333')
    ax.plot(angles_r, values_closed, color=color, linewidth=2.5, label=name, alpha=0.9, marker='s', markersize=5)
    ax.fill(angles_r, values_closed, color=color, alpha=0.10)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles_r[:-1])
ax.set_xticklabels(isaac_seed_labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
ax.set_title("Isaac Robustness Across Seeds", va='bottom', fontsize=14, fontweight='bold', pad=25)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=10, frameon=True, ncol=3)

plt.tight_layout()
plt.savefig('/home/ahmed/Desktop/radar_isaac_robustness.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# 3. TRANSFER ADAPTABILITY RADAR
# ============================================================================
print("Generating Transfer Adaptability Radar...")

adapt_labels = ['Zero-Shot TSR', 'PFT TSR', 'TSR Gain', 'Zero-Shot Jain', 'Reward Eff', 'Adaptability']

# TSR Gain: Max=2.4pp, Adaptability = TSR Gain / 2.4
# Reward Eff: Max=53.33
adapt_data = {
    'MAPPO': [92.1/92.1, 92.5/92.5, 0.4/2.4, 0.6292/0.6510, 53.33/53.33, 0.4/2.4],
    'HiGAT NI-F1F2F3': [76.1/92.1, 78.5/92.5, 2.4/2.4, 0.6510/0.6510, 40.89/53.33, 2.4/2.4],
    'HiGAT no_fl': [75.7/92.1, 77.2/92.5, 1.5/2.4, 0.6448/0.6510, 39.47/53.33, 1.5/2.4],
}

num_vars_a = len(adapt_labels)
angles_a = np.linspace(0, 2 * np.pi, num_vars_a, endpoint=False).tolist()
angles_a += angles_a[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for name, values in adapt_data.items():
    values_closed = values + values[:1]
    color = COLORS.get(name.lower().split()[0], '#333333')
    ax.plot(angles_a, values_closed, color=color, linewidth=2.5, label=name, alpha=0.9, marker='^', markersize=6)
    ax.fill(angles_a, values_closed, color=color, alpha=0.08)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles_a[:-1])
ax.set_xticklabels(adapt_labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8, color='gray')
ax.set_title("Isaac Transfer Adaptability", va='bottom', fontsize=14, fontweight='bold', pad=25)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=10, frameon=True, ncol=3)

plt.tight_layout()
plt.savefig('/home/ahmed/Desktop/radar_isaac_adaptability.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# 4. CROSS-CONDITION OBJECTIVE RADAR (IID vs NONIID)
# ============================================================================
print("Generating Cross-Condition Objective Radar...")

cross_labels = ['TSR', 'Fairness', 'EnergyEff', 'StepRew', 'Throughput', 'FairStab', 'Entropy', 'Stability']

# HiGAT-HAPPO in IID vs NONIID
cross_data = {
    'HiGAT-HAPPO (IID)': [0.97, 0.99, 0.70, 0.72, 0.82, 0.72, 0.55, 0.79],
    'HiGAT-HAPPO (NONIID)': [0.91, 0.97, 0.95, 0.75, 0.92, 0.46, 0.96, 1.00],
    'MAPPO (IID)': [0.96, 0.98, 0.45, 0.79, 0.81, 0.58, 0.80, 0.49],
    'MAPPO (NONIID)': [0.95, 0.94, 0.55, 0.81, 0.97, 0.35, 0.72, 0.31],
}

num_vars_c = len(cross_labels)
angles_c = np.linspace(0, 2 * np.pi, num_vars_c, endpoint=False).tolist()
angles_c += angles_c[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

cross_colors = {'HiGAT-HAPPO (IID)': '#1A237E', 'HiGAT-HAPPO (NONIID)': '#5C6BC0', 
                'MAPPO (IID)': '#1B5E20', 'MAPPO (NONIID)': '#66BB6A'}
cross_styles = {'HiGAT-HAPPO (IID)': '-', 'HiGAT-HAPPO (NONIID)': '--', 
                 'MAPPO (IID)': '-', 'MAPPO (NONIID)': '--'}

for name, values in cross_data.items():
    values_closed = values + values[:1]
    ax.plot(angles_c, values_closed, color=cross_colors[name], linewidth=2.5, 
            label=name, alpha=0.9, linestyle=cross_styles[name])
    ax.fill(angles_c, values_closed, color=cross_colors[name], alpha=0.03)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles_c[:-1])
ax.set_xticklabels(cross_labels, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("Cross-Condition Objective Comparison", va='bottom', fontsize=14, fontweight='bold', pad=25)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=9, frameon=True, ncol=2)

plt.tight_layout()
plt.savefig('/home/ahmed/Desktop/radar_cross_condition.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("\n=== NEW RADAR CHARTS GENERATED ===")
