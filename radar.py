#!/usr/bin/env python3
"""
Radar chart animation
Thin wrapper that delegates to mstd package + scripts/radar.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from scripts.radar import main

if __name__ == "__main__":
    main()
