#!/bin/bash
#SBATCH --time=0:45:00
#SBATCH --mem=8G

module load anaconda gcc
source activate dpvim
export PYTHONUNBUFFERED=1
BASE_FOLDER="" # set to root level for outputs

n=$SLURM_ARRAY_TASK_ID
run_param_file="params_adult.txt"
epsilon=`sed -n "${n} p" ${run_param_file} | awk '{print $1}'`
seed=`sed -n "${n} p" ${run_param_file} | awk '{print $2}'`

variant="all"
init_auto_scale=1.0

output_dir=$BASE_FOLDER/dpvim/adult/results/
log_dir=$BASE_FOLDER/dpvim/adult/logs/
log_path=$log_dir/task_number_%A_%a.out
mkdir -p $output_dir $log_dir

srun --output=$log_path python main_adult.py $variant --epsilon=${epsilon} --seed=${seed} --init_auto_scale=${init_auto_scale} --output_path="$output_dir"
