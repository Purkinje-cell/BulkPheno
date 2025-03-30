#!/bin/bash

# Input directory containing .h5ad files
INPUT_DIR="../data/MERFISH"

# Get list of .h5ad files
FILES=($INPUT_DIR/*.h5ad)
NUM_JOBS=${#FILES[@]}

# Update run script with actual number of jobs
sed -i "s/{MAX_JOBS}/$NUM_JOBS/" run_preprocess.sh

# Submit array job
sbatch \
    --partition=cn-short \
    --account=zeminz_g1 \
    --qos=zeminzcns \
    --job-name=ydc_preprocess \
    --cpus-per-task=20 \
    --array=1-$NUM_JOBS \
    --output=./log/%j_%x_%a.out \
    run_preprocess.sh

echo "Submitted $NUM_JOBS preprocessing jobs"
