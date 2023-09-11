#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=8G

set -e # stop script if any command errors

module load anaconda gcc
source activate dpvim
export PYTHONUNBUFFERED=1

commit_sha=$(git rev-parse --short HEAD)
echo "Commit of code repo ${commit_sha}"

params_file=nondp_run_params.txt
n=$SLURM_ARRAY_TASK_ID
infer_seed=`sed -n "${n} p" $params_file | awk '{print $1}'`
n_epochs=`sed -n "${n} p" $params_file | awk '{print $2}'`

## Parameters to set manually
save_traces_flag="--save_traces"

output_dir="./results/nondp/" # the directory to store the output
mkdir -p $output_dir

############## Run twinify
numpyro_model_path="model_adjusted.py"
log_file_dir="./logs/"
mkdir -p $log_file_dir

input_data_path="./USCensus1990.data.txt"
srun --output "${log_file_dir}/task_number_%A_%a.out" python infer_nondp_baseline.py $input_data_path $numpyro_model_path $output_dir --seed=$infer_seed --k=16 --num_epochs=$n_epochs $save_traces_flag
