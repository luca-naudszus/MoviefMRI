#!/usr/bin/env python3
"""
Denoise fMRIPrep-preprocessed BOLD data with confound regression (nilearn). 

For each preprocessed run, the script
(1) loads the matching confounds timeseries using a named denoising strategy.
(2) reads the TR from the JSON sidecar
(3) regresses the confounds out within fMRIPrep's brain mask
(4) writes a BIDS-derivative-style output plus a JSON sidecar describing exactly what was done,
and a QC table.

Requirements: pip install nilearn

"""
import argparse
import json
import re
import sys
from pathlib import Path

import nilearn
import numpy as np
import pandas as pd

# Denoising strategies
STRATEGIES: dict[str, dict] = {
    "simple": dict(
        strategy=("motion", "wm_csf", "high_pass"),
        motion = "full", wm_csf="basic",
    )
    #TODO: potentially add different strategies
}

# Locate files
#TODO: locate files

# Denoise a single run
def denoise():
    #TODO: define function
    return

# CLI
def parse_args(argv = None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    #TODO: add arguments

def main(argv=None):
    args = parse_args(argv)
    #TODO: add main
