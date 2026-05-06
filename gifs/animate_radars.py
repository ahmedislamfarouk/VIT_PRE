import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Data from previous research
iid_labels = ['TSR', 'Fairness', 'EnergyEff', 'StepRew', 'Throughput', 'FairStab', 'Entropy', 'Stability']
iid_data = {
    'HiGAT-HAPPO (Full)': [0.97, 0.99, 0.70, 0.72, 0.82, 0.72, 0.55, 0.79],
    'MAPPO (No FL)':      [0.96, 0.98, 0.45, 0.79, 0.81, 0.58, 0.80, 0.49],
    'MAT (No FL)':        [1.00, 0.96, 0.05, 0.03, 0.14, 0.72, 0.00, 0.00],
    'GAT-NoEdge-HAPPO':   [0.95, 0.99, 0.67, 0.67, 0.80, 0.61, 0.54, 0.79],
    'FlatGAT-HAPPO':      [0.95, 0.99, 0.60, 0.61, 0.76, 0.66, 0.54, 0.89],
    'HAPPO (No FL)':      [1.00, 0.88, 0.01, 0.05, 0.08, 0.12, 0.00, 0.00],
}

noniid_top_data = {
    'NI: F2 Only (Energy)':  [0.98, 1.00, 0.85, 0.73, 0.87, 0.82, 0.97, 1.00],
    'NI: F2+F3 (Ene+Fair)':  [0.97, 0.99, 0.89, 0.74, 0.91, 0.75, 0.97, 0.46],
    'NI: F1+F3 (Rew+Fair)':  [0.96, 0.99, 0.95, 0.78, 0.93, 0.65, 0.97, 0.60],
    'MAPPO (No FL)':         [0.95, 0.94, 0.55, 0.81, 0.97, 0.35, 0.72, 0.31],
    'HiGAT-HAPPO (Full)':    [0.91, 0.97, 0.95, 0.75, 0.92, 0.46, 0.96, 1.00],
}

COLORS = ['#1A237E', '#2E7D32', '#E65100', '#C62828', '#6A1B9A', '#37474F']

def create_animated_radar(data_dict, labels, filename, title):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Pre-setup axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['', '', '', '', '1.0'], alpha=0) # Clean look
    
    lines = []
    fills = []
    names = list(data_dict.keys())
    
    for i, name in enumerate(names):
        line, = ax.plot([], [], color=COLORS[i % len(COLORS)], linewidth=2, label=name, alpha=0.8)
        fill = ax.fill([], [], color=COLORS[i % len(COLORS)], alpha=0.05)[0]
        lines.append(line)
        fills.append(fill)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')

    def update(frame):
        # frame goes from 0 to 100
        factor = frame / 100.0
        # Easing function for smoothness (Out-Cubic)
        factor = 1 - (1 - factor)**3
        
        for i, (name, values) in enumerate(data_dict.items()):
            v = [val * factor for val in values]
            v += v[:1]
            lines[i].set_data(angles, v)
            # Update fill by creating a new path or modifying vertices
            fills[i].set_xy(np.column_stack([angles, v]))
        return lines + fills

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 100, 60), blit=True)
    
    print(f"Saving {filename}...")
    ani.save(filename, writer=PillowWriter(fps=30))
    plt.close()

# Generate both
create_animated_radar(iid_data, iid_labels, '/home/ahmed/Desktop/radar_iid_growth.gif', "IID Algorithm Performance Growth")
create_animated_radar(noniid_top_data, iid_labels, '/home/ahmed/Desktop/radar_noniid_growth.gif', "NONIID Top Performers Growth")

print("Done!")
