#!/bin/bash
#SBATCH --partition=cn-short
#SBATCH --account=zeminz_g1
#SBATCH --qos=zeminzcns
#SBATCH --job-name=ydc_preprocess
#SBATCH --cpus-per-task=20
#SBATCH --output=./log/%j_%x_%a.out
#SBATCH --array=1-{MAX_JOBS}

# Load environment
source /lustre2/zeminz_pkuhpc/zhangwenjie/neoadjuvant/WES/bashrc_ydc
conda activate perm

# Input directory containing .h5ad files
INPUT_DIR="../data/MERFISH"
FILES=($INPUT_DIR/*.h5ad)

# Get specific file for this array task
FILE=${FILES[$SLURM_ARRAY_TASK_ID-1]}

# Run preprocessing for this file
echo "Processing $FILE"
python preprocess.py --input "$FILE" --output_dir "../data/processed" --hops 3
