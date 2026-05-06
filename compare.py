#!/usr/bin/env python3
"""
compare_backbones.py — Backbone Comparison
Thin wrapper that delegates to mstd package + scripts/compare.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from scripts.compare import main

if __name__ == "__main__":
    main()
