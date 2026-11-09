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
#TODO: Add logging
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import nibabel as nib
from nilearn.image import clean_img, smooth_img
from nilearn.interfaces.fmriprep import load_confounds

# Denoising strategies
STRATEGIES: dict[str, dict] = {
    "simple": dict(
        strategy=("motion", "wm_csf", "high_pass"),
        motion = "full", wm_csf="basic",
    )
    #TODO: potentially add different strategies
}

# Locate files
def find_runs(fmriprep_dir: Path, space: str, participants: list[str],
              tasks: list[str], sessions: list[str]):
    pattern = f"*_space-{space}_desc-preproc_bold.nii*"
    runs = sorted(fmriprep_dir.glob(f"sub-*/**/func/{pattern}"))
    keep = []
    for r in runs:
        files = dict.re.findall(r"([a-ZA-Z]+)-([a-zA-Z0-9]+)", r.name.split(".")[0])
        if participants and r.get("sub") not in participants:
            continue
        if tasks and r.get("task") not in tasks:
            continue
        if sessions and r.get("session") not in sessions:
            continue
        keep.append(r)
    return keep

def get_tr(bold: Path):
    sidecar = replace_fn(bold, "_bold", "bold.json")
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        if "RepetitionTime" in meta:
            return meta["RepetitionTime"]
    tr = float(nib.load(str(bold)).header.get_zooms()[3])
    return tr

def get_mask(bold: Path):
    desc = re.search(r"_desc-[a-zA-Z0-9]+_bold", bold.name)
    mask = replace_fn(bold, desc, "_desc-brain_mask.nii.gz")
    return mask

def replace_fn(bold: Path, old: str, new: str):
    name = bold.name
    for extension in (".nii.gz", ".nii"):
        if name.endswith(extension):
            name = name[: -len(extension)]
            break
    return bold.with_name(name.replace(old, new))

# Denoise a single run
def denoise_run(bold: Path, fmriprep_dir: Path, out_dir: Path, strategy: str,
                kwargs: dict, smoothing: float | None, detrend: bool,
                standardize: bool, low_pass: float, high_pass: float):
    bold_rel = bold.relative_to(fmriprep_dir)
    desc = re.search(r"_desc-[a-zA-Z0-9]+_bold", bold.name) 
    out_file = out_dir / bold_rel.parent / bold.name.replace(desc, "_desc_denoised_bold")
    qc = dict(run=bold.name, output=str(out_file))

    tr = get_tr(bold)
    mask = get_mask(bold)

    confounds, sample_mask = load_confounds(str(bold), kwargs)

    n_vols = len(confounds)
    n_kept = n_vols if sample_mask is None else len(sample_mask)
    conf_tsv = replace_fn(bold, desc, "_desc-confounds_timeseries.tsv")

    fd = pd.read_csv(conf_tsv, sep="\t")["framewise_discplacement"]
    qc.update(tr=tr, n_volumes=n_vols, n_kept=n_kept,
              n_regressors=confounds.shape[1],
              mean_fd=round(float(fd.mean()), 3))

    img = bold
    if smoothing:
        img = smooth_img(bold, fwhm=smoothing)

    cleaned = clean_img(img, confounds=confounds, detrend=detrend,
                        standardize=standardize, low_pass=low_pass, high_pass=high_pass,
                        t_r = tr, mask_img = mask, clean__sample_mask = sample_mask,
    )

    out_file.parent.mkdir()
    # reset header because nilearn resets the 4th zoom
    cleaned.header.set_zooms(cleaned.header.get_zooms()[:3] + (tr,))
    cleaned.header.set_xyzt_units("mm", "sec")
    cleaned.to_filename(out_file)


    confounds.to_csv(replace_fn(out_file, "_desc-denoised_bold",
                                "_desc-denoised_regressors.tsv"), sep = "\t", index=False)

    sidecar = dict(
        Sources=[str(bold_rel)],
        RepetitionTime=tr,
        DenoisingStratgy=strategy,
        LoadConfoundsParameters={k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()},
        Regressors=confounds.columns,
        LowPass=low_pass,
        HighPass=high_pass,
        SmoothingFWHM=smoothing,
        Standardize=standardize,
        NumberOfVolumesOriginal=n_vols,
        NumberOfVolumesRetained=n_kept,
        RetainedVolumeIndices=(None if sample_mask is None else np.asarray(sample_mask).tolist()),
    )
    replace_fn(out_file, "_bold", "_bold.json").write_text(
        json.dumps(sidecar, indent=2)
    )

    return qc

# CLI
def parse_args(argv = None):
    #TODO: Add help and defaults, error message?
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("fmriprep_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--strategy", default = "simple")
    p.add_argument("--space", default = "MNI152NLin2009cAsym")
    p.add_argument("--participant")
    p.add_argument("--session")
    p.add_argument("--task")
    p.add_argument("--low-pass")
    p.add_argument("--high-pass")
    p.add_argument("--smoothing")
    p.add_argument("--standardize")
    args = p.parse_args(argv)
    return args

def main(argv=None):
    args = parse_args(argv)
    kwargs = dict(STRATEGIES[args.strategy])
    fmriprep_dir = args.fmriprepdir
    out_dir = args.output_dir / args.strategy
    out_dir.mkdir()
    runs = find_runs(fmriprep_dir, args.space, args.participant, args.task, args.session)

    for r in runs:
        out = denoise_run(r, fmriprep_dir, out_dir, args.strategy, kwargs,
                        args.low_pass, args.high_pass, args.smoothing, args.standardize,
                        args.max_censored, args.overwrite)
        out_df = pd.DataFrame(out)
        out_df.to_csv(out_dir / "denoising.tsv", sep="\t", index=False)
        #TODO: Write dataset description.json

    #TODO: Write error control
    
    return
