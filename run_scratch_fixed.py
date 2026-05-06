#!/usr/bin/env python3
"""
Fixed version with proper model configs for SigLIP and DINOv2
Thin wrapper that delegates to mstd package + scripts/run_scratch.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from scripts.run_scratch import main

if __name__ == "__main__":
    main()
