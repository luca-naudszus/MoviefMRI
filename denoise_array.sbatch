#!/bin/bash
# One SLURM array task per participant. 
#
# Before submitting:
#   1. make the participant list (one label per line, without "sub-"):
#        ls -d /data/derivatives/fmriprep/sub-*/ | xargs -n1 basename \
#            | sed 's/^sub-//' > participants.txt
#   2. create the log folder:  mkdir -p logs
#   3. submit with the array size matching the list (%20 = max 20 at once):
#        sbatch --array=1-$(wc -l < participants.txt)%20 denoise_array.sbatch
#
#SBATCH --job-name=denoise
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/denoise_%A_%a.out

set -euo pipefail

FMRIPREP_DIR=/data/derivatives/fmriprep
OUT_DIR=/data/derivatives/denoised
SCRIPT=$HOME/code/denoise_fmriprep.py

module load python

SUB=$(sed -n "${SLURM_ARRAY_TASK_ID}p" participants.txt)
echo "Array task ${SLURM_ARRAY_TASK_ID}: sub-${SUB}"

python "$SCRIPT" "$FMRIPREP_DIR" "$OUT_DIR" \
    --participant "$SUB" \
    --task rest \
    --strategy scrubbing \
    --global-signal-full \
    --low-pass 0.1 \
    --n-jobs "${SLURM_CPUS_PER_TASK}"
