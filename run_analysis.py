#!/usr/bin/env python3
"""Run the full time series forecasting pipeline.

Usage:
    python run_analysis.py
"""

# macOS threading fix — must be set before importing numpy/tensorflow
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore')

from pipeline import run_pipeline

if __name__ == '__main__':
    run_pipeline()
