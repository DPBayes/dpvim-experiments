#!/bin/bash
#SBATCH --time=0:45:00
#SBATCH --mem=8G

module load anaconda gcc
source activate dpvim
export PYTHONUNBUFFERED=1

n=$SLURM_ARRAY_TASK_ID
run_param_file="params_adult.txt"
epsilon=`sed -n "${n} p" ${run_param_file} | awk '{print $1}'`
seed=`sed -n "${n} p" ${run_param_file} | awk '{print $2}'`

variant="all"
init_auto_scale=1.0

srun python main_adult.py ${variant} --epsilon=${epsilon} --seed=${seed} --init_auto_scale=${init_auto_scale} --output_path="./results/" --adjusted_regression
