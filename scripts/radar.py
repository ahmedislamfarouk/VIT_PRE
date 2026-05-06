#!/usr/bin/env python3
"""
Radar chart animation for model performance comparison.
"""
import sys

from mstd.visualization.gif import create_radar_animation


def main():
    deit_target = [0.9127, 0.9141, 0.9151, 0.9144]
    dinov2_target = [0.9493, 0.9502, 0.9506, 0.9504]
    siglip2_target = [0.8817, 0.8857, 0.8858, 0.8848]

    output_path = '/home/skyvision/ammarbigass5/results/smooth_sequential_comparison.gif'
    if len(sys.argv) > 1:
        output_path = sys.argv[1]

    create_radar_animation(output_path, deit_target, dinov2_target, siglip2_target)


if __name__ == "__main__":
    main()
