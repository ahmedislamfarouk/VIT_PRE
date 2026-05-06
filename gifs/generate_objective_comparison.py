import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# --- Data Definitions ---
conditions = ['IID Condition', 'NONIID Condition']
data = {
    'MAPPO (Baseline)': [0.6888, 0.7666],
    'Rank 1 (Optimized)': [0.7420, 0.8835]
}

COLORS = ['#2E7D32', '#1A237E']
FINAL_DIR = '/home/ahmed/Desktop/gifs/final_comparison'
FPS, FRAMES_PER_ALGO, HOLD_FRAMES = 30, 100, 150 # Very slow for clarity

def create_objective_bar(data_dict, labels, filename, title):
    names = list(data_dict.keys())
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2, left=0.15)
    
    indices = np.arange(len(labels))
    width = 0.35 # Thick bars
    
    rects1 = ax.bar(indices - width/2, [0, 0], width, label=names[0], color='#2E7D32', alpha=0.85, edgecolor='black', linewidth=1.5)
    rects2 = ax.bar(indices + width/2, [0, 0], width, label=names[1], color='#1A237E', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    texts1 = [ax.text(0, 0, '', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#1B5E20') for _ in range(len(labels))]
    texts2 = [ax.text(0, 0, '', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#0D47A1') for _ in range(len(labels))]
    
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
    ax.set_ylabel('Objective Function Score (0-1)', fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12, frameon=True, shadow=True)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    def update(frame):
        for i, (rects, texts, data_key) in enumerate([(rects1, texts1, names[0]), (rects2, texts2, names[1])]):
            sf, ef = i*FRAMES_PER_ALGO, (i+1)*FRAMES_PER_ALGO
            factor = 0 if frame < sf else (1 if frame > ef else 1 - (1 - (frame-sf)/FRAMES_PER_ALGO)**3)
            for rect, text, val in zip(rects, texts, data_dict[data_key]):
                h = val * factor
                rect.set_height(h)
                if h > 0.05:
                    text.set_position((rect.get_x() + rect.get_width()/2, h + 0.02))
                    text.set_text(f'{h:.3f}')
                else: text.set_text('')
        return list(rects1) + list(rects2) + texts1 + texts2

    ani = FuncAnimation(fig, update, frames=2*FRAMES_PER_ALGO + HOLD_FRAMES, blit=True)
    ani.save(os.path.join(FINAL_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()

print("Generating Objective Comparison GIF...")
create_objective_bar(data, conditions, 'final_objective_comparison.gif', "Overall Objective Score Comparison")
print("Done!")
