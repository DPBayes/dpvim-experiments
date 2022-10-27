This folder contains the code used to produce the experiment for full-rank aligned gradients DPVI.

The code was originally ran in a cluster, paralellizing over the random seeds and epsilons. The seeds and epsilons are described in the create_params_adult.py script. The main script main_adult.py is called using the seeds and epsilons as
```
python linear_regression_infer.py all --epsilon=epsilon --seed=seed --corr_method=manual --corr_strength=strength --num_dims=num_dims --run_aligned_full --run_vanilla_full --results_path=path_for_results
```

The plots for Figure 5 are then created by calling
```
python plot_corr_strength_box.py path_for_results --corr_method=manual
```

To generate the list of random seeds and epsilons, run
```
python create_params_linreg.py
```

In case you want to run the full experiment, you can use the `run_experiment_seq.sh` script. **NOTE** this will take a very long while, since it runs all the repeats in sequence.

We will also include the slurm workload manager script, that was used to run the example (see `slurm_linear_regression.sh`).
