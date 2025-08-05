#!/bin/bash

#SBATCH --job-name=job-%j
#SBATCH --output=/users/bdillon/projects/rl/logs/log-%j.txt
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=72:00:00
#SBATCH --mem=36GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.dillon@ulster.ac.uk

RUNNAME="tttt_run1"

if [ -d "/users/bdillon/projects/tinn/results/$RUNNAME" ]; then
    echo "Directory '$RUNNAME' already exists. Exiting."
    exit 1  # Exit with error status
fi

# if directory doesn't exist, create it and continue
mkdir -p "/users/bdillon/projects/tinn/results/$RUNNAME"
echo "Directory 'tinn/results/$RUNNAME' created."

module load python3
#module load amd-rocm/rocm-6.3.3
#source /users/bdillon/venvs/ml-rocm/bin/activate
source /users/bdillon/venvs/ml/bin/activate

CONFIG_FILE='configs/tttt_params1.yml'

python /users/bdillon/projects/tinn/rl/run.py --config "$CONFIG_FILE" --runname "$RUNNAME" --save_path "/users/bdillon/projects/tinn/results/$RUNNAME"

