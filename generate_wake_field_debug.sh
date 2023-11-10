#!/bin/bash
#SBATCH --nodes=2
#SBATCH --time=01:00:00
#SBATCH --ntasks=10
#SBATCH --job-name=generate_wake_field
#SBATCH --output=generate_wake_field_debug.%j.out

module purge
# module load anaconda
# source /curc/sw/anaconda3/2020.11/etc/profile.d/conda.sh
module load anaconda
conda activate rl_wf_env_rocm

cd /projects/aohe7145/projects/nn_wake_modeling/
python WakeField.py
