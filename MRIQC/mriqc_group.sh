#!/bin/bash
#SBATCH --job-name=mriqc_group
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2:00:00

source /cluster/work/gess/sbs/MoviefMRI/code/environment.sh
BIDS=/cluster/work/gess/sbs/MoviefMRI/data
IMAGE=/cluster/work/gess/sbs/neurodesk/neurocommand/local/containers/mriqc_24.0.2_20241108/mriqc_24.0.2_20241108.simg
OUT=$BIDS/derivatives/mriqc
mkdir -p $SCRATCH/mriqc_work/group/tmp
apptainer run --cleanenv --bind $SCRATCH/mriqc_work/group/tmp:/tmp \
    $IMAGE mriqc $BIDS $OUT group -w $SCRATCH/mriqc_work/group --verbose-reports

