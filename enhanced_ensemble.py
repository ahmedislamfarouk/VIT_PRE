#!/usr/bin/env python3
"""
Enhanced Multi-Backbone Ensemble
Thin wrapper that delegates to mstd package + scripts/enhanced_ensemble.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from scripts.enhanced_ensemble import main

if __name__ == "__main__":
    main()
