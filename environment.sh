#!/usr/bin/env bash
# environment.sh
# Pins the Neurodesk tool versions used for this project.
# Usage: source environment.sh


set -euo pipefail

# Start from a clean slate so no stray modules from a previous session leak in
module purge

# Make Neurodesk modules available (necessary?)
# module load neurodesk/use

# --- Pinned tool versions -----------------------------------------
module load datalad/1.6.2
module load fmriprep/25.2.5
module load mriqc/24.0.2

# --------------------------------------------------------------------
# Set variables
export FS_LICENSE=$HOME/FSLicense/license.txt

# -------------------------------------------------------------------
module list
