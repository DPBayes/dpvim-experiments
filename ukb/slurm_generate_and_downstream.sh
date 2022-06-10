#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=8G

set -e # stop script if any command errors

source activate dpvim
export PYTHONUNBUFFERED=1

n=$SLURM_ARRAY_TASK_ID
infer_seed=`sed -n "${n} p" dp_run_params.txt | awk '{print $1}'` # Get seed
eps=`sed -n "${n} p" dp_run_params.txt | awk '{print $2}'` # Get epsilon

##############
## Parameters to set manually
n_epochs=1000
clipping_threshold=2.0
prefix="vanilla"
num_synthetic_data_sets=100
avg_over_epochs=200

############## Paths
result_dir="" # the directory in which inferred model params are stored
downstream_result_dir="" # the directory in which downstream analysis results will be stored
numpyro_model_path="model1_wholepop.py"
input_data_path=""

##############
srun python generate_and_downstream.py $input_data_path $numpyro_model_path $result_dir --output_dir=$downstream_result_dir --epsilon=$eps --seed=$infer_seed --k=16 --num_epochs=$n_epochs --clipping_threshold=$clipping_threshold --prefix=$prefix --num_synthetic_data_sets=$num_synthetic_data_sets --avg_over=$avg_over_epochs
