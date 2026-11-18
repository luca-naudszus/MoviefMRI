#!/usr/bin/env bash
# versions.sh
# Pins all tool versions and shared paths of this project. 
#
# This file only defines variables: no module loads, no other side effects. It is
#   - sourced by environment.sh (interactive work), or
#   - read directly by the *_sub.py submit scripts (no need to source anything first).

# --- Tool versions ------------------------------------------------------------
DATALAD_VERSION=1.6.2
FMRIPREP_VERSION=25.2.5
FMRIPREP_BUILD=20260413        
MRIQC_VERSION=24.0.2
MRIQC_BUILD=20241108

# --- Container images (derived from the versions above) -----------------------
CONTAINERS=/cluster/work/gess/sbs/neurodesk/neurocommand/local/containers
export FMRIPREP_IMAGE=$CONTAINERS/fmriprep_${FMRIPREP_VERSION}_${FMRIPREP_BUILD}/fmriprep_${FMRIPREP_VERSION}_${FMRIPREP_BUILD}.simg
export MRIQC_IMAGE=$CONTAINERS/mriqc_${MRIQC_VERSION}_${MRIQC_BUILD}/mriqc_${MRIQC_VERSION}_${MRIQC_BUILD}.simg

# --- Python environment for denoise.py ----------------------------------------
# The package versions are pinned in code/requirements.txt.
STACK_MODULE=stack/2025-06          
PYTHON_MODULE=python/3.13.0
export DENOISE_VENV=/cluster/work/gess/sbs/MoviefMRI/venv

# --- Shared resources ---------------------------------------------------------
export FS_LICENSE=$HOME/FSLicense/license.txt
export TEMPLATEFLOW_HOME=/cluster/work/gess/sbs/MoviefMRI/templateflow