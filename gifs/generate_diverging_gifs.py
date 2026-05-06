import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# --- Data Definitions ---
labels = ['TSR', 'Fairness', 'EnergyEff', 'StepRew', 'Throughput', 'FairStab', 'Entropy', 'Stability']

iid_data = {
    'MAPPO (Baseline)':   [0.96, 0.98, 0.45, 0.79, 0.81, 0.58, 0.80, 0.49],
    'HiGAT-HAPPO':        [0.97, 0.99, 0.70, 0.72, 0.82, 0.72, 0.55, 0.79],
    'MAT':                [1.00, 0.96, 0.05, 0.03, 0.14, 0.72, 0.00, 0.00],
    'GAT-NoEdge-HAPPO':   [0.95, 0.99, 0.67, 0.67, 0.80, 0.61, 0.54, 0.79],
}

noniid_data = {
    'MAPPO (Baseline)':   [0.95, 0.94, 0.55, 0.81, 0.97, 0.35, 0.72, 0.31],
    'NI: F2 Only (Energy)': [0.98, 1.00, 0.85, 0.73, 0.87, 0.82, 0.97, 1.00],
    'NI: F2+F3 (Ene+Fair)': [0.97, 0.99, 0.89, 0.74, 0.91, 0.75, 0.97, 0.46],
    'HiGAT-HAPPO':        [0.91, 0.97, 0.95, 0.75, 0.92, 0.46, 0.96, 1.00],
}

COLORS_POS = '#2E7D32' # Green for better
COLORS_NEG = '#C62828' # Red for worse
TWEAKED_DIR = '/home/ahmed/Desktop/gifs/tweaked'
FPS, FRAMES_PER_ALGO, HOLD_FRAMES = 30, 60, 90

def create_diverging_delta_chart(data_dict, labels, filename, title):
    names = list(data_dict.keys())
    baseline_name = names[0] # MAPPO is first
    baseline_vals = np.array(data_dict[baseline_name])
    compare_names = names[1:]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(left=0.25, bottom=0.15)
    
    indices = np.arange(len(labels))
    width = 0.8 / len(compare_names)
    
    rects_list = []
    text_list = []
    
    for i, name in enumerate(compare_names):
        # Create bars at 0
        rects = ax.barh(indices + (i - len(compare_names)/2 + 0.5) * width, [0]*len(labels), width, 
                        label=f'Δ {name}', alpha=0.8)
        rects_list.append(rects)
        algo_texts = [ax.text(0, 0, '', va='center', fontsize=8, fontweight='bold') for _ in range(len(labels))]
        text_list.append(algo_texts)

    ax.axvline(0, color='black', linewidth=1.5) # Central axis
    ax.set_yticks(indices)
    ax.set_yticklabels(labels, fontweight='bold')
    ax.set_xlabel('Delta from MAPPO (Positive = Better, Negative = Worse)', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(-0.6, 0.6) # Standardized scale for deltas
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=True)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    def update(frame):
        for i, name in enumerate(compare_names):
            deltas = np.array(data_dict[name]) - baseline_vals
            sf, ef = i*FRAMES_PER_ALGO, (i+1)*FRAMES_PER_ALGO
            factor = 0 if frame < sf else (1 if frame > ef else 1 - (1 - (frame-sf)/FRAMES_PER_ALGO)**3)
            
            for rect, text, d_val in zip(rects_list[i], text_list[i], deltas):
                current_d = d_val * factor
                rect.set_width(current_d)
                rect.set_color(COLORS_POS if current_d >= 0 else COLORS_NEG)
                
                if abs(current_d) > 0.01:
                    x_pos = current_d + (0.01 if current_d >= 0 else -0.01)
                    text.set_position((x_pos, rect.get_y() + rect.get_height()/2))
                    text.set_text(f'{current_d:+.2f}')
                    text.set_ha('left' if current_d >= 0 else 'right')
                else:
                    text.set_text('')
        return [r for rects in rects_list for r in rects] + [t for texts in text_list for t in texts]

    ani = FuncAnimation(fig, update, frames=len(compare_names)*FRAMES_PER_ALGO + HOLD_FRAMES, blit=True)
    ani.save(os.path.join(TWEAKED_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()

# Generate Diverging Delta Charts
print("Generating Diverging IID Delta chart...")
create_diverging_delta_chart(iid_data, labels, 'bar_diverging_iid_delta.gif', "IID Performance Delta (Relative to MAPPO)")

print("Generating Diverging NONIID Delta chart...")
create_diverging_delta_chart(noniid_data, labels, 'bar_diverging_noniid_delta.gif', "NONIID Performance Delta (Relative to MAPPO)")

# --- Special Cross-Env Comparison (Diverging IID vs NONIID) ---
def create_cross_env_diverging(algo_name, iid_vals, niid_vals, labels, filename):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.15)
    
    indices = np.arange(len(labels))
    ax.axvline(0, color='black', linewidth=2)
    
    # Left bars (IID - inverted)
    rects_iid = ax.barh(indices, [0]*len(labels), 0.4, color='#1A237E', label='IID Performance', alpha=0.8)
    # Right bars (NONIID)
    rects_niid = ax.barh(indices, [0]*len(labels), 0.4, color='#5C6BC0', label='NONIID Performance', alpha=0.8)
    
    ax.set_yticks(indices)
    ax.set_yticklabels(labels, fontweight='bold')
    ax.set_xlim(-1.1, 1.1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(['1.0', '0.5', '0', '0.5', '1.0']) # Absolute values for both sides
    ax.set_title(f"Environment Shift: {algo_name}\n(Left: IID | Right: NONIID)", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    def update(frame):
        factor = frame / 100.0
        factor = 1 - (1 - factor)**3
        for rect, val in zip(rects_iid, iid_vals): rect.set_width(-val * factor)
        for rect, val in zip(rects_niid, niid_vals): rect.set_width(val * factor)
        return list(rects_iid) + list(rects_niid)

    ani = FuncAnimation(fig, update, frames=100 + HOLD_FRAMES, blit=True)
    ani.save(os.path.join(TWEAKED_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()

print("Generating Cross-Env Diverging chart for HiGAT...")
create_cross_env_diverging("HiGAT-HAPPO", iid_data['HiGAT-HAPPO'], noniid_data['HiGAT-HAPPO'], labels, 'bar_comparison_iid_noniid.gif')

print("All diverging animations completed!")
