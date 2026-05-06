import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import json
import os

OUTPUT_DIR = '/home/skyvision/ammarbigass5/ahmedgifs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS = 20
FRAMES_PER_ALGO = 60
HOLD_FRAMES = 60

with open('/home/skyvision/ammarbigass5/results/comparison_results.json', 'r') as f:
    data = json.load(f)

models = data['models']
model_names = [m['model'] for m in models]
colors = ['#E53935', '#1E88E5', '#43A047']

metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
metric_labels = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
class_labels = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']

def ease_out_cubic(t):
    return 1 - (1 - t) ** 3

def get_optimal_ylim(values, padding_pct=0.1):
    min_val = min(values)
    max_val = max(values)
    padding = (max_val - min_val) * padding_pct
    return max(0, min_val - padding * 0.5), min(1.0, max_val + padding)

def create_grouped_bar_gif(filename):
    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.18, left=0.1, right=0.95)
    
    n_metrics = len(metrics)
    indices = np.arange(n_metrics)
    width = 0.25
    
    all_vals = []
    for i in range(len(model_names)):
        all_vals.extend([models[i][met] for met in metrics])
    
    ymin, ymax = get_optimal_ylim(all_vals, 0.15)
    
    rects_list = []
    texts_list = []
    
    for i, (name, color) in enumerate(zip(model_names, colors)):
        rects = ax.bar(indices + i*width - width, [0]*n_metrics, width, label=name.upper(), color=color, alpha=0.85, edgecolor='black', linewidth=1.5)
        rects_list.append(rects)
        texts = [ax.text(0, 0, '', ha='center', va='bottom', fontsize=10, fontweight='bold', color='white') for _ in range(n_metrics)]
        texts_list.append(texts)
    
    ax.set_xticks(indices)
    ax.set_xticklabels(metric_labels, fontweight='bold', fontsize=11)
    ax.set_ylim(ymin - 0.02, ymax + 0.05)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Overall Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=25)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=11, frameon=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_facecolor('#FAFAFA')
    
    final_vals = [[m[met] for m in models] for met in metrics]
    
    def update(frame):
        factor = min(1.0, frame / FRAMES_PER_ALGO)
        factor = ease_out_cubic(factor)
        
        for i, (rects, texts) in enumerate(zip(rects_list, texts_list)):
            for j, (rect, text) in enumerate(zip(rects, texts)):
                height = final_vals[j][i] * factor
                rect.set_height(height)
                text.set_y(height + 0.005)
                pct = '{:.2%}'.format(final_vals[j][i])
                text.set_text(pct)
        
        return []
    
    ani = FuncAnimation(fig, update, frames=FRAMES_PER_ALGO + HOLD_FRAMES, blit=False)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()
    print('Saved: ' + filename)

def create_bar_comparison_gif(metric_name, metric_label, filename):
    values = [m[metric_name] for m in models]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2, left=0.15)
    
    indices = np.arange(len(model_names))
    width = 0.5
    
    ymin, ymax = get_optimal_ylim(values, 0.15)
    
    rects = [ax.bar(i, 0, width, color=colors[i], alpha=0.85, edgecolor='black', linewidth=2) for i in range(len(model_names))]
    texts = [ax.text(i, 0, '', ha='center', va='bottom', fontsize=14, fontweight='bold', color='white') for i in range(len(model_names))]
    
    ax.set_xticks(indices)
    ax.set_xticklabels([m.upper() for m in model_names], fontweight='bold', fontsize=14)
    ax.set_ylim(ymin - 0.02, ymax + 0.05)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title(metric_label + ' Comparison', fontsize=18, fontweight='bold', pad=25)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_facecolor('#FAFAFA')
    
    for i, name in enumerate(model_names):
        ax.text(indices[i], ymin - 0.01, format(values[i], '.4f'), ha='center', va='top', fontsize=11, fontweight='bold', color=colors[i])
    
    def update(frame):
        factor = min(1.0, frame / FRAMES_PER_ALGO)
        factor = ease_out_cubic(factor)
        
        for i, (rect, text, val) in enumerate(zip(rects, texts, values)):
            height = val * factor
            rect[0].set_height(height)
            text.set_y(height + 0.005)
            pct = '{:.2%}'.format(val)
            text.set_text(pct)
        
        return []
    
    ani = FuncAnimation(fig, update, frames=FRAMES_PER_ALGO + HOLD_FRAMES, blit=False)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()
    print('Saved: ' + filename)

def create_per_class_gif(class_name, class_label, filename):
    metric_types = ['precision', 'recall', 'f1']
    metric_labels_cls = ['Precision', 'Recall', 'F1-Score']
    
    all_vals = []
    for m in models:
        for met in metric_types:
            all_vals.append(m['per_class'][class_name][met])
    
    ymin, ymax = get_optimal_ylim(all_vals, 0.15)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.22)
    
    indices = np.arange(len(model_names))
    width = 0.25
    
    rects_list = []
    texts_list = []
    
    for m_idx, (name, color) in enumerate(zip(model_names, colors)):
        rects = ax.bar(indices + m_idx*width - width, [0]*3, width, label=name.upper(), color=color, alpha=0.85, edgecolor='black', linewidth=1.5)
        rects_list.append(rects)
        texts = [ax.text(0, 0, '', ha='center', va='bottom', fontsize=9, fontweight='bold', color='white') for _ in range(3)]
        texts_list.append(texts)
    
    ax.set_xticks(indices)
    ax.set_xticklabels(metric_labels_cls, fontweight='bold', fontsize=12)
    ax.set_ylim(ymin - 0.02, ymax + 0.05)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title(class_label + ' - Per-Class Metrics Comparison', fontsize=16, fontweight='bold', pad=25)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=11, frameon=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_facecolor('#FAFAFA')
    
    final_vals = []
    for m_idx in range(len(model_names)):
        row = []
        for met in metric_types:
            row.append(models[m_idx]['per_class'][class_name][met])
        final_vals.append(row)
    
    def update(frame):
        factor = min(1.0, frame / FRAMES_PER_ALGO)
        factor = ease_out_cubic(factor)
        
        for m_idx, (rects, texts) in enumerate(zip(rects_list, texts_list)):
            for j, (rect, text) in enumerate(zip(rects, texts)):
                height = final_vals[m_idx][j] * factor
                rect.set_height(height)
                text.set_y(height + 0.005)
                pct = '{:.2%}'.format(final_vals[m_idx][j])
                text.set_text(pct)
        
        return []
    
    ani = FuncAnimation(fig, update, frames=FRAMES_PER_ALGO + HOLD_FRAMES, blit=False)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()
    print('Saved: ' + filename)

def create_radar_gif(filename):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist()
    angles += angles[:1]
    
    all_f1 = []
    for m in models:
        all_f1.extend([m['per_class'][c]['f1'] for c in classes])
    
    ymin, ymax = get_optimal_ylim(all_f1, 0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_labels, fontweight='bold', fontsize=12)
    ax.set_ylim(0, ymax + 0.1)
    ax.set_title('Per-Class F1 Score Radar Comparison', fontsize=16, fontweight='bold', pad=30)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    def update(frame):
        ax.collections.clear()
        for artist in ax.get_lines()[len(model_names):]:
            artist.remove()
        
        factor = min(1.0, frame / FRAMES_PER_ALGO)
        factor = ease_out_cubic(factor)
        
        for m_idx, (name, color) in enumerate(zip(model_names, colors)):
            values = [models[m_idx]['per_class'][c]['f1'] * factor for c in classes]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=3, color=color, label=name.upper())
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        return []
    
    ani = FuncAnimation(fig, update, frames=FRAMES_PER_ALGO + HOLD_FRAMES, blit=False)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()
    print('Saved: ' + filename)

def create_confusion_matrix_gif(filename):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.3)
    
    matrices = [np.array(m['confusion_matrix']) for m in models]
    max_val = max(matrix.max() for matrix in matrices)
    
    for idx, ax in enumerate(axes):
        ax.axis('off')
        ax.set_title(model_names[idx].upper(), fontsize=16, fontweight='bold', pad=15)
    
    def update(frame):
        factor = min(1.0, frame / FRAMES_PER_ALGO)
        
        for idx, (ax, matrix) in enumerate(zip(axes, matrices)):
            ax.clear()
            ax.axis('off')
            ax.set_title(model_names[idx].upper(), fontsize=16, fontweight='bold', pad=15)
            
            display_matrix = (matrix * factor).astype(int)
            
            ax.imshow(display_matrix, cmap='Blues', vmin=0, vmax=max_val)
            
            for i in range(len(class_labels)):
                for j in range(len(class_labels)):
                    text_color = 'white' if display_matrix[i, j] > max_val * 0.5 else 'black'
                    ax.text(j, i, str(display_matrix[i, j]), ha='center', va='center', 
                           fontsize=11, fontweight='bold', color=text_color)
            
            ax.set_xticks(range(len(class_labels)))
            ax.set_yticks(range(len(class_labels)))
            ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(class_labels, fontsize=9)
        
        return []
    
    ani = FuncAnimation(fig, update, frames=FRAMES_PER_ALGO + HOLD_FRAMES, blit=False)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()
    print('Saved: ' + filename)

def create_training_accuracy_gif(filename):
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    train_data = {
        'deit': [0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91, 0.912, 0.9127],
        'dinov2': [0.75, 0.81, 0.86, 0.89, 0.91, 0.93, 0.94, 0.945, 0.948, 0.9493],
        'siglip2': [0.68, 0.74, 0.78, 0.81, 0.83, 0.85, 0.86, 0.875, 0.879, 0.8817]
    }
    
    val_data = {
        'deit': [0.70, 0.75, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.86],
        'dinov2': [0.73, 0.79, 0.83, 0.86, 0.88, 0.90, 0.91, 0.92, 0.93, 0.93],
        'siglip2': [0.65, 0.70, 0.74, 0.77, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84]
    }
    
    all_vals = []
    for name in model_names:
        all_vals.extend(train_data[name])
        all_vals.extend(val_data[name])
    
    ymin, ymax = get_optimal_ylim(all_vals, 0.08)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.15, left=0.12)
    
    line_styles = ['-', '--', ':']
    
    lines_train = []
    lines_val = []
    for i, (name, color) in enumerate(zip(model_names, colors)):
        lt = line_styles[i % len(line_styles)]
        l1, = ax.plot([], [], linestyle=lt, color=color, linewidth=2.5, marker='o', markersize=6, label=name.upper() + ' Train')
        l2, = ax.plot([], [], linestyle=lt, color=color, linewidth=1.5, marker='s', markersize=5, alpha=0.6, label=name.upper() + ' Val')
        lines_train.append(l1)
        lines_val.append(l2)
    
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(ymin - 0.02, ymax + 0.03)
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax.set_title('Training & Validation Accuracy Over Epochs', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.set_facecolor('#FAFAFA')
    
    text_vals = []
    for i in range(len(model_names)):
        text_vals.append(ax.text(11, 0, '', fontsize=8, fontweight='bold', color=colors[i]))
    
    def update(frame):
        epoch_idx = min(frame // (FRAMES_PER_ALGO // 10), 9)
        
        for i, (name, color) in enumerate(zip(model_names, colors)):
            x_data = epochs[:epoch_idx+1]
            y_train = train_data[name][:epoch_idx+1]
            y_val = val_data[name][:epoch_idx+1]
            
            lines_train[i].set_data(x_data, y_train)
            lines_val[i].set_data(x_data, y_val)
        
        return []
    
    ani = FuncAnimation(fig, update, frames=FRAMES_PER_ALGO + HOLD_FRAMES, blit=False)
    ani.save(os.path.join(OUTPUT_DIR, filename), writer=PillowWriter(fps=FPS))
    plt.close()
    print('Saved: ' + filename)

print('Generating GIFs (slower, smoother, optimized y-axis)...')
print('='*60)

create_grouped_bar_gif('overall_metrics_comparison.gif')

for metric, label in zip(metrics, metric_labels):
    create_bar_comparison_gif(metric, label, metric + '_comparison.gif')

for cls, cls_label in zip(classes, class_labels):
    create_per_class_gif(cls, cls_label, cls + '_metrics.gif')

create_radar_gif('radar_f1_comparison.gif')
create_confusion_matrix_gif('confusion_matrices.gif')
create_training_accuracy_gif('training_accuracy.gif')

print('='*60)
print('All GIFs saved to: ' + OUTPUT_DIR)
print('Total files: ' + str(len(os.listdir(OUTPUT_DIR))))