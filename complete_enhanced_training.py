#!/usr/bin/env python3
"""
complete_enhanced_training.py — Finish the AMTD final training
Thin wrapper that delegates to mstd package + scripts/complete_enhanced_training.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from scripts.complete_enhanced_training import main

if __name__ == "__main__":
    main()
