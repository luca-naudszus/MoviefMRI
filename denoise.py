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
def find_runs():
    #TODO: locate files
    return

# Denoise a single run
def denoise_run():
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
    strategy = args.strategy
    strategy_args = None #TODO: how do I get strategy arguments?
    fmriprep_dir = args.fmriprepdir
    out_dir = args.output_dir / args.strategy
    out_dir.mkdir()
    runs = find_runs(fmriprep.dir, args.space, args.participant, args.task, args.session)

    for r in runs:
        out = denoise_run(r, fmriprep_dir, out_dir, args.strategy, strategy_args,
                        args.low_pass, args.high_pass, args.smoothing, args.standardize,
                        args.max_censored, args.overwrite)
        out_df = pd.DataFrame(out)
        out_df.to_csv(out_dir / "denoising.tsv", sep="\t", index=False)
        #TODO: Write dataset description.json
    
    return
