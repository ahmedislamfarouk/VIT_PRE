#!/usr/bin/env python3
"""
evaluate_data_hunger.py — Data-Efficiency Evaluation
Thin wrapper that delegates to mstd package + scripts/evaluate_data_hunger.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from scripts.evaluate_data_hunger import main

if __name__ == "__main__":
    main()
