#!/bin/bash -l
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null

set -e # stop script if any command errors

module load anaconda gcc
source activate dpvim_gpu
export PYTHONUNBUFFERED=1

BASE_FOLDER="" # set to root level for outputs

commit_sha=$(git rev-parse --short HEAD)
echo "Commit of code repo ${commit_sha}"

params_file=params_linreg_toy_manual.txt
n=$SLURM_ARRAY_TASK_ID
infer_seed=`sed -n "${n} p" $params_file | awk '{print $1}'`
eps=`sed -n "${n} p" $params_file | awk '{print $2}'`
corr_strength=`sed -n "${n} p" $params_file | awk '{print $3}'`
num_dims=`sed -n "${n} p" $params_file | awk '{print $4}'`

clipping_threshold="2.0"
n_epochs=1000


output_dir="${BASE_FOLDER}/dpvim/linreg_fullrank/toy_data/manual/"
mkdir -p $output_dir

log_file_dir="${BASE_FOLDER}/dpvim/linreg_fullrank/toy_data/logs/"
mkdir -p $log_file_dir

############## Run inference

srun --output "${log_file_dir}/task_number_%A_%a.out" python linear_regression_infer.py --results_path=$output_dir --epsilon=$eps --seed=$infer_seed --num_epochs=$n_epochs --clipping_threshold=$clipping_threshold --corr_strength=$corr_strength --num_dims=$num_dims --run_aligned_full --run_vanilla_full--corr_method="manual"
