#!/bin/bash
# One SLURM array task per participant. 
#
# Before submitting:
#   1. make the participant list:
#        ls -d /cluster/work/gess/sbs/MoviefMRI/data/derivatives/fmriprep/sub-*/ | xargs -n1 basename \
#            | sed 's/^sub-//' > participants.txt
#   2. create the log folder:  mkdir -p logs
#   3. submit with the array size matching the list (%20 = max 20 at once):
#        sbatch --array=1-$(wc -l < participants.txt)%20 denoise_array.sbatch
#
#SBATCH --job-name=denoise
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/work/gess/sbs/MoviefMRI/code/logs/denoise_%A_%a.out

FMRIPREP_DIR=/cluster/work/gess/sbs/MoviefMRI/data/derivatives/fmriprep
OUT_DIR=/cluster/work/gess/sbs/MoviefMRI/data/derivatives/denoised
SCRIPT=/cluster/work/gess/sbs/MoviefMRI/code/denoise.py

source /cluster/work/gess/sbs/MoviefMRI/code/versions.sh
module load "$STACK_MODULE" "$PYTOHN_MODULE"
set -euo pipefail

SUB=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /cluster/work/gess/sbs/MoviefMRI/code/participants.txt)
echo "Array task ${SLURM_ARRAY_TASK_ID}: sub-${SUB}"

"$DENOISE_VENV/bin/python" "$SCRIPT" "$FMRIPREP_DIR" "$OUT_DIR" \
    --participant "$SUB" \
    --task Rest \
    --strategy compcor \
    --low-pass 0.1 \
    --high-pass 0.008 \
    --n-jobs "${SLURM_CPUS_PER_TASK}"
