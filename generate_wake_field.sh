#!/bin/bash
#SBATCH --nodes=24
#SBATCH --time=08:00:00
#SBATCH --ntasks=500
#SBATCH --job-name=generate_wake_field
#SBATCH --output=generate_wake_field.%j.out

module purge
# module load anaconda
conda activate rl_wf_env_rocm

cd /projects/aohe7145/projects/nn_wake_modeling/
python WakeField.py
