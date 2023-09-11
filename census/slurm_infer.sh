#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --mem=8G

set -e # stop script if any command errors

module load anaconda gcc
source activate dpvim
export PYTHONUNBUFFERED=1

commit_sha=$(git rev-parse --short HEAD)
echo "Commit of code repo ${commit_sha}"

params_file=dp_run_params.txt
n=$SLURM_ARRAY_TASK_ID
infer_seed=`sed -n "${n} p" $params_file | awk '{print $1}'`
eps=`sed -n "${n} p" $params_file | awk '{print $2}'`
dpvi_alg=`sed -n "${n} p" $params_file | awk '{print $3}'`
clipping_threshold=`sed -n "${n} p" $params_file | awk '{print $4}'`
init_scale=`sed -n "${n} p" $params_file | awk '{print $5}'`
n_epochs=`sed -n "${n} p" $params_file | awk '{print $6}'`
sampling_ratio=0.01


## Parameters to set manually
save_traces_flag="--save_traces"

output_dir="./results/init${init_scale}/" # the directory to store the output
mkdir -p $output_dir

############## Run twinify
numpyro_model_path="model_adjusted.py"
log_file_dir="./logs/"
mkdir -p $log_file_dir

input_data_path="./USCensus1990.data.txt"
srun --output "${log_file_dir}/task_number_%A_%a.out" python infer.py $input_data_path $numpyro_model_path $output_dir --dpvi_flavour="${dpvi_alg}" --epsilon=$eps --seed=$infer_seed --k=16 --num_epochs=$n_epochs --clipping_threshold=$clipping_threshold $save_traces_flag --init_scale=$init_scale --sampling_ratio=$sampling_ratio
