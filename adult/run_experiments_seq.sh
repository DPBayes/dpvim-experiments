#!/bin/bash

# create parameters
python create_params_adult.py

# set run params
variant="all"
init_auto_scale=1.0
num_epochs=4000

# loop over the parameter file
while read p; do
    epsilon=$(echo $p | awk '{print $1}')
    seed=$(echo $p | awk '{print $2}')
    python main_adult.py ${variant} --epsilon=${epsilon} --seed=${seed} --init_auto_scale=${init_auto_scale} --output_path="./results/" --adjusted_regression --num_epochs=${num_epochs}
done <params_adult.txt

# plot results 
python plot_accuracy.py --init_auto_scale=${init_auto_scale} --num_epochs=${num_epochs}
python plot_adult_noise.py --init_auto_scale=${init_auto_scale} --num_epochs=${num_epochs}
