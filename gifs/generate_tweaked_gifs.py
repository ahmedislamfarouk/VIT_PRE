import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# --- Data Definitions ---
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

isaac_robustness = {
    'MAPPO': [1.0, 0.94, 0.96, 0.99, 0.90, 0.91],
    'HiGAT NI-F1F2F3': [0.84, 0.79, 0.73, 0.99, 0.94, 0.72],
    'HiGAT no_fl': [0.71, 0.65, 0.46, 1.0, 0.81, 0.94],
}

isaac_adaptability = {
    'MAPPO': [1.0, 1.0, 0.16, 0.96, 1.0, 0.16],
    'HiGAT NI-F1F2F3': [0.82, 0.85, 1.0, 1.0, 0.76, 1.0],
    'HiGAT no_fl': [0.82, 0.83, 0.62, 0.99, 0.74, 0.62],
}

COLORS = ['#1A237E', '#2E7D32', '#E65100', '#C62828', '#6A1B9A', '#37474F']
TWEAKED_DIR = '/home/ahmed/Desktop/gifs/tweaked'
FPS = 30
FRAMES_PER_ALGO = 60
HOLD_FRAMES = 90

def get_area(values):
    # Rough estimate of polygon area: sum of adjacent product
    v = list(values) + [values[0]]
    return sum(v[i] * v[i+1] for i in range(len(values)))

def create_tweaked_radar(data_dict, labels, filename, title):
    # Sort data by area (smallest to largest)
    sorted_names = sorted(data_dict.keys(), key=lambda k: get_area(data_dict[k]))
    sorted_data = {name: data_dict[name] for name in sorted_names}
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(9, 11), subplot_kw=dict(polar=True))
    plt.subplots_adjust(bottom=0.2, top=0.9)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.1)
    
    lines, fills = [], []
    for i, name in enumerate(sorted_names):
        l, = ax.plot([], [], color=COLORS[i % len(COLORS)], linewidth=3, label=name, alpha=0.85)
        f = ax.fill([], [], color=COLORS[i % len(COLORS)], alpha=0.15)[0]
        lines.append(l); fills.append(f)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2 if len(sorted_names)>3 else 1, fontsize=9, frameon=True)
    ax.set_title(title + "\n(Sorted by Surface Area)", pad=40, fontsize=14, fontweight='bold')
    
    def update(frame):
        for i, name in enumerate(sorted_names):
            values = sorted_data[name]
            sf, ef = i*FRAMES_PER_ALGO, (i+1)*FRAMES_PER_ALGO
            factor = 0 if frame < sf else (1 if frame > ef else 1 - (1 - (frame-sf)/FRAMES_PER_ALGO)**3)
            v = [val * factor for val in values] + [values[0] * factor]
            lines[i].set_data(angles, v)
            fills[i].set_xy(np.column_stack([angles, v]))
        return lines + fills

    ani = FuncAnimation(fig, update, frames=len(sorted_names)*FRAMES_PER_ALGO + HOLD_FRAMES, blit=True)
    ani.save(os.path.join(TWEAKED_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()

def create_tweaked_bar(data_dict, labels, filename, title):
    names = list(data_dict.keys())
    num_algos, num_metrics = len(names), len(labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.25)
    
    indices = np.arange(num_metrics)
    width = 0.85 / num_algos # Thicker bars (was 0.8)
    rects_list = []
    text_list = [] # For values on bars
    
    for i, name in enumerate(names):
        rects = ax.bar(indices + (i - num_algos/2 + 0.5) * width, [0]*num_metrics, width, 
                       label=name, color=COLORS[i % len(COLORS)], alpha=0.85, edgecolor='black', linewidth=0.5)
        rects_list.append(rects)
        # Placeholder texts
        algo_texts = [ax.text(0, 0, '', ha='center', va='bottom', fontsize=7, fontweight='bold') for _ in range(num_metrics)]
        text_list.append(algo_texts)

    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontweight='bold')
    ax.set_ylim(0, 1.25) # Extra space for labels
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    def update(frame):
        for i, (name, values) in enumerate(data_dict.items()):
            sf, ef = i*FRAMES_PER_ALGO, (i+1)*FRAMES_PER_ALGO
            factor = 0 if frame < sf else (1 if frame > ef else 1 - (1 - (frame-sf)/FRAMES_PER_ALGO)**3)
            for rect, text, val in zip(rects_list[i], text_list[i], values):
                h = val * factor
                rect.set_height(h)
                if h > 0.05: # Only show text if bar is visible
                    text.set_position((rect.get_x() + rect.get_width()/2, h))
                    text.set_text(f'{h:.2f}')
                else:
                    text.set_text('')
        return [r for rects in rects_list for r in rects] + [t for texts in text_list for t in texts]

    ani = FuncAnimation(fig, update, frames=num_algos*FRAMES_PER_ALGO + HOLD_FRAMES, blit=True)
    ani.save(os.path.join(TWEAKED_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()

datasets = [
    (iid_data, iid_labels, 'iid', "IID Comparison"),
    (noniid_top_data, iid_labels, 'noniid', "NONIID Top Performers"),
    (isaac_robustness, ['S42T', 'S123T', 'S456T', 'S42J', 'S123J', 'S456J'], 'isaac_robust', "Isaac Robustness"),
    (isaac_adaptability, ['ZS-T', 'PFT-T', 'Gain', 'ZS-J', 'Rew', 'Adapt'], 'isaac_adapt', "Isaac Adaptability")
]

for data, labels, key, title in datasets:
    print(f"Generating tweaked {title}...")
    create_tweaked_radar(data, labels, f'radar_{key}_sorted.gif', title)
    create_tweaked_bar(data, labels, f'bar_vertical_{key}_thicker.gif', title)

print("All tweaked animations completed in Desktop/gifs/tweaked!")
