#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=8G

set -e # stop script if any command errors

source activate dpvim
source ../paths.sh
export PYTHONUNBUFFERED=1

n=$SLURM_ARRAY_TASK_ID
infer_seed=`sed -n "${n} p" dp_run_params.txt | awk '{print $1}'` # Get seed
eps=`sed -n "${n} p" dp_run_params.txt | awk '{print $2}'` # Get epsilon

##############
## Parameters to set manually
n_epochs=1000
clipping_threshold=2.0
save_traces_flag="--save_traces"
init_scale="0.1"

############## Paths
output_dir=""
numpyro_model_path="model1_wholepop.py"

input_data_path=""
srun python main_aligned.py $input_data_path $numpyro_model_path $output_dir --epsilon=$eps --seed=$infer_seed --k=16 --num_epochs=$n_epochs --clipping_threshold=$clipping_threshold $save_traces_flag --adjusted_regression --init_scale=$init_scale
