import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# --- Data Definitions ---
labels = ['TSR', 'Fairness', 'EnergyEff', 'StepRew', 'Throughput', 'FairStab', 'Entropy', 'Stability']

iid_comp = {
    'MAPPO (Baseline)': [0.96, 0.98, 0.45, 0.79, 0.81, 0.58, 0.80, 0.49],
    'HiGAT-HAPPO':      [0.97, 0.99, 0.70, 0.72, 0.82, 0.72, 0.55, 0.79]
}

noniid_comp = {
    'MAPPO (Baseline)':   [0.95, 0.94, 0.55, 0.81, 0.97, 0.35, 0.72, 0.31],
    'NI: F2 Only (Rank 1)': [0.98, 1.00, 0.85, 0.73, 0.87, 0.82, 0.97, 1.00]
}

COLORS = ['#2E7D32', '#1A237E'] # MAPPO Green, HiGAT/NI Blue
FINAL_DIR = '/home/ahmed/Desktop/gifs/final_comparison'
FPS, FRAMES_PER_ALGO, HOLD_FRAMES = 30, 80, 120 # Even slower!

def create_diverging(data_dict, filename, title):
    names = list(data_dict.keys())
    base_vals = np.array(data_dict[names[0]])
    comp_vals = np.array(data_dict[names[1]])
    deltas = comp_vals - base_vals
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.25, bottom=0.15)
    
    indices = np.arange(len(labels))
    ax.axvline(0, color='black', linewidth=2)
    
    # Single set of bars for the delta
    rects = ax.barh(indices, [0]*len(labels), 0.6, color='gray', alpha=0.9, edgecolor='black')
    texts = [ax.text(0, 0, '', va='center', fontsize=10, fontweight='bold') for _ in range(len(labels))]
    
    ax.set_yticks(indices)
    ax.set_yticklabels(labels, fontweight='bold', fontsize=11)
    ax.set_xlim(-0.6, 0.6)
    ax.set_title(f"Performance Gap: {names[1]} vs {names[0]}\n(Diverging Delta Chart)", fontsize=14, fontweight='bold', pad=25)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    def update(frame):
        factor = min(1.0, frame / FRAMES_PER_ALGO)
        factor = 1 - (1 - factor)**3
        for rect, text, d in zip(rects, texts, deltas):
            curr_d = d * factor
            rect.set_width(curr_d)
            rect.set_color('#2E7D32' if curr_d >= 0 else '#C62828')
            if abs(curr_d) > 0.01:
                text.set_position((curr_d + (0.02 if curr_d >= 0 else -0.02), rect.get_y() + rect.get_height()/2))
                text.set_text(f'{curr_d:+.2f}')
                text.set_ha('left' if curr_d >= 0 else 'right')
        return list(rects) + texts

    ani = FuncAnimation(fig, update, frames=FRAMES_PER_ALGO + HOLD_FRAMES, blit=True)
    ani.save(os.path.join(FINAL_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()

def create_thick_bar(data_dict, filename, title):
    names = list(data_dict.keys())
    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.2)
    
    indices = np.arange(len(labels))
    width = 0.4 # EXTREMELY THICK BARS
    
    rects1 = ax.bar(indices - width/2, [0]*len(labels), width, label=names[0], color='#2E7D32', alpha=0.8, edgecolor='black', linewidth=1.5)
    rects2 = ax.bar(indices + width/2, [0]*len(labels), width, label=names[1], color='#1A237E', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    texts1 = [ax.text(0, 0, '', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1B5E20') for _ in range(len(labels))]
    texts2 = [ax.text(0, 0, '', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0D47A1') for _ in range(len(labels))]
    
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=0, fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=12, frameon=True, shadow=True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    def update(frame):
        # Sequential: Algo 1 then Algo 2
        for i, (rects, texts, data_key) in enumerate([(rects1, texts1, names[0]), (rects2, texts2, names[1])]):
            sf, ef = i*FRAMES_PER_ALGO, (i+1)*FRAMES_PER_ALGO
            factor = 0 if frame < sf else (1 if frame > ef else 1 - (1 - (frame-sf)/FRAMES_PER_ALGO)**3)
            for rect, text, val in zip(rects, texts, data_dict[data_key]):
                h = val * factor
                rect.set_height(h)
                if h > 0.05:
                    text.set_position((rect.get_x() + rect.get_width()/2, h + 0.01))
                    text.set_text(f'{h*100:.0f}%' if val > 0.8 else f'{h:.2f}')
                else: text.set_text('')
        return list(rects1) + list(rects2) + texts1 + texts2

    ani = FuncAnimation(fig, update, frames=2*FRAMES_PER_ALGO + HOLD_FRAMES, blit=True)
    ani.save(os.path.join(FINAL_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()

print("Generating Final Comparison GIFs...")
create_diverging(iid_comp, 'final_diverging_iid.gif', "IID Comparison: HiGAT vs MAPPO Delta")
create_diverging(noniid_comp, 'final_diverging_noniid.gif', "NONIID Comparison: NI-F2 vs MAPPO Delta")
create_thick_bar(iid_comp, 'final_bar_thick_iid.gif', "IID Results: Detailed Metric Comparison")
create_thick_bar(noniid_comp, 'final_bar_thick_noniid.gif', "NONIID Results: Detailed Metric Comparison")
print("Done!")
