#!/usr/bin/env python3
"""
Denoise fMRIPrep-preprocessed BOLD data with confound regression (nilearn). 

For each preprocessed run, the script
(1) loads one of nilearn's strategies, optionally adjusted
(2) reads the TR from the JSON sidecar
(3) regresses the confounds out within fMRIPrep's brain mask
(4) writes a BIDS-derivative-style output plus a JSON sidecar describing exactly what was done,
and a QC table.

Requirements: pip install nilearn

Preset nilearn definitions and which options each one accepts:
https://nilearn.github.io/stable/modules/generated/nilearn.interfaces.fmriprep.load_confounds_strategy.html

"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import nibabel as nib
from nilearn.image import clean_img, smooth_img
import nilearn
from nilearn.interfaces.fmriprep import load_confounds_strategy

# Presets and overrides
# simple:       24 motion parameters, WM/CSF means, cosine drift regressors
# scrubbing:    as simple with 8 WM/CSF parameters, 
#               plus censoring of volumes with FD > 0.55 mm or std DVARS > 1.5 (and segments < 5 volumes)
# compcor:      24 motion parameters, anatomical CompCor, cosines
# ica_aroma:    #TODO: uses slightly different outputs, currently not implemented
PRESETS = {"simple", "scrubbing", "compcor"} #, "ica_aroma"}
OVERRIDES = {
    "motion": "motion",
    "wm_csf": "wm_csf",
    "global_signal": "global_signal",
    "fd_threshold": "fd_threshold",
    "dvars_threshold": "dvars_threshold",
}

# Logger
log = logging.getLogger("denoise")

# Locate files
def entities(path: Path): 
    return dict(re.findall(r"([a-ZA-Z]+)-([a-zA-Z0-9]+)", path.name.split(".")[0]))

def find_runs(fmriprep_dir: Path, space: str, participants: list[str],
              tasks: list[str], sessions: list[str]):
    pattern = f"*_space-{space}_desc-preproc_bold.nii*"
    runs = sorted(fmriprep_dir.glob(f"sub-*/**/func/{pattern}"))
    keep = []
    for r in runs:
        files = entities(r)
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
            return float(meta["RepetitionTime"])
    tr = float(nib.load(str(bold)).header.get_zooms()[3])
    log.warning("%s: no repetition time in sidecar, using header TR=%.3f s", bold.name, tr)
    return tr

def get_mask(bold: Path):
    desc = re.search(r"_desc-[a-zA-Z0-9]+_bold", bold.name)
    mask = replace_fn(bold, desc, "_desc-brain_mask.nii.gz")
    return mask if mask.exists() else None

def replace_fn(bold: Path, old: str, new: str):
    name = bold.name
    for extension in (".nii.gz", ".nii"):
        if name.endswith(extension):
            name = name[: -len(extension)]
            break
    return bold.with_name(name.replace(old, new))

# Denoise a single run
def denoise_run(bold: Path, fmriprep_dir: Path, out_dir: Path, strategy: str,
                overrides: dict, smoothing: float | None, 
                standardize: bool, low_pass: float, high_pass: float,
                max_censored: float, overwrite: bool)-> dict:
    
    bold_rel = bold.relative_to(fmriprep_dir) 
    desc = re.search(r"_desc-[a-zA-Z0-9]+_bold", bold.name).group(0)
    out_file = out_dir / bold_rel.parent / bold.name.replace(desc, "_desc_denoised_bold")
    out_file = out_file.with_name(out_file.name.replace(".nii.gz", "").replace(".nii", "") + ".nii.gz")
    qc = dict(participant=entities(bold)["sub"],
              run=bold.name, output=str(out_file), status="ok")

    if out_file.exists() and not overwrite:
        log.info("skip (exists): %s", out_file.name)
        qc["status"] = "skipped_exists"
        return qc

    try:
        tr = get_tr(bold)
        mask = get_mask(bold)
        if mask is None:
            log.warning("%s: no brain mask found, cleaning all voxels", bold.name)

        confounds, sample_mask = load_confounds_strategy(str(bold), denoise_strategy=strategy, **overrides)

        n_vols = len(confounds)
        n_kept = n_vols if sample_mask is None else len(sample_mask)
        frac_censored = 1 - n_kept / n_vols
        conf_tsv = replace_fn(bold, desc, "_desc-confounds_timeseries.tsv")

        fd = pd.read_csv(conf_tsv, sep="\t")["framewise_discplacement"]
        qc.update(tr=tr, n_volumes=n_vols, n_kept=n_kept,
                  pct_censored=round(100*frac_censored, 1),
                    n_regressors=confounds.shape[1],
                    dof_left=n_kept - confounds.shape[1],
                    mean_fd=round(float(fd.mean()), 3))

        if frac_censored > max_censored:
            log.warning("%s: %.0f%% of volumes censored (> %.0f%%), skipped", 
                        bold.name, 100 * frac_censored, 100 * max_censored)
            qc["status"] = "excluded_motion"
            return qc
        if qc["dof_left"] < 1:
            raise ValueError(f"more regressors ({confounds.shape[1]}) than retained volumes ({n_kept})")

        img = bold
        if smoothing:
            img = smooth_img(str(bold), fwhm=smoothing)

        cleaned = clean_img(img, confounds=confounds, detrend=False,
                            standardize=standardize, low_pass=low_pass, high_pass=high_pass,
                            t_r = tr, mask_img = mask, clean__sample_mask = sample_mask,
        )

        out_file.parent.mkdir(parents=True, exist_ok=True)
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
            StrategyOverrides=overrides,
            NilearnVersion=nilearn.__version__,
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
        log.info("done: %s (%d regressors, %d/%d volumes kept)",
                 out_file.name, confounds.shape[1], n_kept, n_vols)
    except Exception as e:
        log.error("%s failed: %s", bold.name, e)
        qc.update(status="error", error=str(e))
    return qc

# QC tables
def tmp_write(path: Path, text: str):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    
def write_qc(out_dir: Path, qc_df: pd.DataFrame):
    for sub, rows in qc_df.groupby("Participant"):
        f = out_dir / f"sub-{sub}" / f"sub-{sub}_desc-denoising_qc.tsv"
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists():
            old = pd.read_csv(f, sep="\t", dtype={"participant": str})
            new = rows[~((rows["status"] == "skipped_exists") & rows["run"].isin(old["run"]))]
            rows = pd.concat([old[~old["run"].isin(new["run"])], new])
        tmp_write(f, rows.sort_values("run").convert_dtypes().to_csv(sep="\t", index=False))
    parts = [pd.read_csv(f, sep="\t", dtype={"participant": str}) 
             for f in sorted(out_dir.glob("sub-*/sub-*_desc_denoising_qc.tsv"))]
    tmp_write(out_dir / "denoising_qc.tsv", 
              pd.concat(parts).convert_dtypes().to_csv(sep="\t", index=False))

# CLI
def parse_args(argv = None):
    #TODO: Add help? 
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("fmriprep_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--strategy", default = "simple", choices=PRESETS)
    p.add_argument("--global-signal", choices=["basic", "power2", "derivatives", "full"])
    p.add_argument("--wm-csf", choices=["basic", "power2", "derivatives", "full"])
    p.add_argument("--motion", choices=["basic", "power2", "derivatives", "full"])
    p.add_argument("--space", default = "MNI152NLin2009cAsym")
    p.add_argument("--participant", nargs="+")
    p.add_argument("--session", nargs="+")
    p.add_argument("--task", nargs="+")
    p.add_argument("--low-pass", type=float, default=None)
    p.add_argument("--high-pass", type=float, default=None)
    p.add_argument("--smoothing", type=float, default=None)
    p.add_argument("--standardize", choices=["zscore_sample", "psc"], default=None)
    p.add_argument("--max_censored", type=float, default=0.5)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    overrides = {kw: getattr(args, opt) for opt, kw in OVERRIDES.items() if getattr(args, opt) is not None}
    label = "_".join([args.strategy] + [f"{k}--{v}" for k, v in sorted(overrides.items())])

    fmriprep_dir = args.fmriprepdir.resolve()
    out_dir = args.output_dir.resolve() / label


    runs = find_runs(fmriprep_dir, args.space, args.participant, args.task, args.session)
    if not runs:
        log.error("No matching runs in %s (space=%s).", fmriprep_dir, args.space)
        return 1

    log.info("Strategy '%s' %s: %d run(s) to process", args.strategy, overrides or "(preset defaults)", len(runs))

    qc = Parallel(n_jobs=args.n_jobs)(
    delayed(denoise_run(r, fmriprep_dir, out_dir, args.strategy, overrides,
                    args.low_pass, args.high_pass, args.smoothing, args.standardize,
                    args.max_censored, args.overwrite))
                    for r in runs)
    qc_df = pd.DataFrame(qc)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_qc(out_dir, qc_df)
    tmp_write(out_dir / "dataset_description.json", json.dumps(dict(
            Name = f"fMRIPrep denoised",
            BIDSVersion="1.9.0",
            DatasetType="derivative",
            GeneratedBy=[dict(Name="denoise.py (nilearn)")],
        ), indent=2))

    counts = qc_df["status"].value_counts().to_dict()
    log.info("Finished: %s QC table: %s", counts, out_dir / "denoising_qc.tsv")
    return 0 if "error" not in counts else 2

if __name__ == "__main__":
    sys.exit(main())
