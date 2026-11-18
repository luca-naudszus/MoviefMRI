#!/bin/bash
#SBATCH --job-name=mriqc_group
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2:00:00

# The MRIQC image is pinned in versions.sh (an absolute path, because Slurm runs a copy of this script from elsewhere)
source /cluster/work/gess/sbs/MoviefMRI/code/versions.sh
if [ ! -f "$MRIQC_IMAGE" ]; then
    echo "ERROR: MRIQC image not found: $MRIQC_IMAGE" >&2
    exit 1
fi
echo "Image: $MRIQC_IMAGE"

BIDS=/cluster/work/gess/sbs/MoviefMRI/data
OUT=$BIDS/derivatives/mriqc
mkdir -p $SCRATCH/mriqc_work/group/tmp
apptainer run --cleanenv --bind $SCRATCH/mriqc_work/group/tmp:/tmp \
    $MRIQC_IMAGE $BIDS $OUT group -w $SCRATCH/mriqc_work/group

