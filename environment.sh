#!/usr/bin/env bash
# environment.sh
# Prepares neurodesk modules for this project for usage in interactive sessions. 
# 
# Usage: source environment.sh


set -euo pipefail

# Start from a clean slate so no stray modules from a previous session leak in
module purge

# Make Neurodesk modules available (necessary?)
# module load neurodesk/use

# --- Pinned tool versions -----------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/versions.sh"
module load datalad/1.6.2
module load fmriprep/25.2.5
module load mriqc/24.0.2

# -------------------------------------------------------------------
module list
