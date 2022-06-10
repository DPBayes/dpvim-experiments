#!/bin/sh

ERROR_TYPE=stderr
EPSILON=1.0

python plot_trace_error.py /scratch/cs/synthetic-data-twins/ukb/processed_data/model_one_covid_tested_data.csv /scratch/cs/synthetic-data-twins/ukb/synthetic_data/poisson_regression/after_purge/adjusted/ /scratch/cs/synthetic-data-twins/ukb/synthetic_data/poisson_regression/after_purge/adjusted/nondp_baselines --output_dir=./ --clipping_threshold=2.0 --adjusted_regression --init_scale=1.0 --epsilon=$EPSILON --num_epochs=200 --plot_error_over_epochs --error_type=$ERROR_TYPE

for INIT_SCALE in "0.01" "0.1" "1.0"
do
    python plot_trace_error.py /scratch/cs/synthetic-data-twins/ukb/processed_data/model_one_covid_tested_data.csv /scratch/cs/synthetic-data-twins/ukb/synthetic_data/poisson_regression/after_purge/adjusted/ /scratch/cs/synthetic-data-twins/ukb/synthetic_data/poisson_regression/after_purge/adjusted/nondp_baselines --output_dir=./ --clipping_threshold=2.0 --adjusted_regression --init_scale=$INIT_SCALE --epsilon=$EPSILON --num_epochs=200 --plot_traces --plot_robustness --error_type=$ERROR_TYPE
done
