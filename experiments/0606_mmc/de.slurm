#!/bin/bash
#SBATCH --job-name=TO-MMC-DE
#SBATCH --mail-user=s3607828@vuw.leidenuniv.nl
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/top_opt_%A_%a.out
#SBATCH --error=logs/top_opt_%A_%a.err
#SBATCH --array=1-15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096MB
#SBATCH --partition=cpu-medium
#SBATCH --time=23:59:00


# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change

module load Python/3.10.4-GCCcore-11.3.0

# activating the environment
source $HOME/envs/de/bin/activate

# Specify the path to the config file
config=$HOME/TO/experiments/0606_mmc/de.tsv

# Read the property from a file based on SLURM_ARRAY_TASK_ID
# id	dimension	optimizer	seed
dimension=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
optimizer=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
seed=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

echo "Running topology optimization with property: $PROPERTY"
# Replace the following line with your actual command
python3 $HOME/TO/experiments/0606_mmc/run.py --dimension dimension --optimizer optimizer --seed seed