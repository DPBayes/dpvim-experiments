#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=8G

set -e # stop script if any command errors

module load anaconda
source activate dpvim
source ../paths.sh
export PYTHONUNBUFFERED=1

BASE_FOLDER="" # set to root level for outputs

commit_sha=$(git rev-parse --short HEAD)
echo "Commit of code repo ${commit_sha}"

n=$SLURM_ARRAY_TASK_ID
params_file="dp_run_params_fig1.txt"
infer_seed=`sed -n "${n} p" $params_file | awk '{print $1}'`
eps=`sed -n "${n} p" $params_file | awk '{print $2}'`
dpvi_alg=`sed -n "${n} p" $params_file | awk '{print $3}'`
clipping_threshold=`sed -n "${n} p" $params_file | awk '{print $4}'`
init_scale=`sed -n "${n} p" $params_file | awk '{print $5}'`
n_epochs=`sed -n "${n} p" $params_file | awk '{print $6}'`

## Parameters to set manually
num_synthetic_data_sets=100
avg_over_epochs=1

inference_results_dir="${BASE_FOLDER}/dpvim/ukb/init${init_scale}/" # the directory in which inferred model params are stored
downstream_results_dir="${BASE_FOLDER}/dpvim/ukb/init${init_scale}/downstream/" # the directory in which downstream analysis results will be stored

mkdir -p $downstream_results_dir

model_path="model.py"
log_file_dir="${BASE_FOLDER}/dpvim/ukb/logs/"
mkdir -p $log_file_dir

input_data_path="${UKB_BASE_FOLDER}/processed_data/model_one_covid_tested_data.csv"

dependency_arg=""
infer_batch_id=$1
if [ ${infer_batch_id} ]; then
    infer_job_id=${infer_batch_id}_${n}
    dependency_arg="--dependency=afterok:${infer_job_id}"
fi
srun ${dependency_arg} --output "${log_file_dir}/task_number_%A_%a.out" python generate_and_downstream.py $input_data_path $model_path $inference_results_dir --output_dir=$downstream_results_dir --epsilon=$eps --seed=$infer_seed --k=16 --num_epochs=$n_epochs --clipping_threshold=$clipping_threshold --dpvi_flavour=$dpvi_alg --num_synthetic_data_sets=$num_synthetic_data_sets --avg_over=$avg_over_epochs
