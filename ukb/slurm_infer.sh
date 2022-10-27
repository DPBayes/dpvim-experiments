#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=8G

set -e # stop script if any command errors

module load anaconda gcc
source activate dpvim
source ../paths.sh
export PYTHONUNBUFFERED=1

BASE_FOLDER="" # set to root level for outputs

commit_sha=$(git rev-parse --short HEAD)
echo "Commit of code repo ${commit_sha}"

params_file=dp_run_params_fig1.txt
n=$SLURM_ARRAY_TASK_ID
infer_seed=`sed -n "${n} p" $params_file | awk '{print $1}'`
eps=`sed -n "${n} p" $params_file | awk '{print $2}'`
dpvi_alg=`sed -n "${n} p" $params_file | awk '{print $3}'`
clipping_threshold=`sed -n "${n} p" $params_file | awk '{print $4}'`
init_scale=`sed -n "${n} p" $params_file | awk '{print $5}'`
n_epochs=`sed -n "${n} p" $params_file | awk '{print $6}'`


## Parameters to set manually
save_traces_flag="--save_traces"

output_dir="${BASE_FOLDER}/dpvim/ukb/init${init_scale}/" # the directory to store the twinify output
mkdir -p $output_dir

############## Run twinify
numpyro_model_path="model.py"
log_file_dir="${BASE_FOLDER}/dpvim/ukb/logs/"
mkdir -p $log_file_dir

input_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data.csv"
srun --output "${log_file_dir}/task_number_%A_%a.out" python infer.py $input_data_path $numpyro_model_path $output_dir --dpvi_flavour=$dpvi_alg --epsilon=$eps --seed=$infer_seed --k=16 --num_epochs=$n_epochs --clipping_threshold=$clipping_threshold $save_traces_flag --init_scale=$init_scale
