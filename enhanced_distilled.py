#!/usr/bin/env python3
"""
enhanced_distilled.py — Adaptive Multi-Teacher Distillation
Thin wrapper that delegates to mstd package + scripts/enhanced_distilled.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from scripts.enhanced_distilled import main

if __name__ == "__main__":
    main()
