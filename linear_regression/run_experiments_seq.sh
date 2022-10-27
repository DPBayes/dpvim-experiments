#!/bin/bash

# create parameters
python create_params_linreg.py

output_dir="./results"
n_epochs=1000
clipping_threshold=2.0

# loop over the parameter file
while read p; do
    seed=$(echo $p | awk '{print $1}')
    epsilon=$(echo $p | awk '{print $2}')
    corr_strength=$(echo $p | awk '{print $3}')
    num_dims=$(echo $p | awk '{print $4}')
    python linear_regression_infer.py --results_path=$output_dir --epsilon=$epsilon --seed=$seed --num_epochs=$n_epochs --clipping_threshold=$clipping_threshold --corr_strength=$corr_strength --num_dims=$num_dims --run_aligned_full --run_vanilla_full --corr_method="manual"

done <params_linreg_toy_manual.txt

# plot results
python plot_corr_strength_box.py $output_dir --corr_method=manual --clipping_threshold=$clipping_threshold
