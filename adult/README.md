This folder contains the code used to produce the Adult experiment.

The code was originally ran in a cluster, paralellizing over the random seeds and epsilons. The seeds and epsilons are described in the create_params_adult.py script. The main script main_adult.py is called using the seeds and epsilons as
```
python main_adult.py all --epsilon=epsilon --seed=seed --init_auto_scale=1.0
```

The plots for Figure 4 are then created by calling
```
python plot_accuracy.py --init_auto_scale=${init_auto_scale} --num_epochs=${num_epochs}
python plot_adult_noise.py --init_auto_scale=${init_auto_scale} --num_epochs=${num_epochs}
```

To generate the list of random seeds and epsilons, run
```
python create_params_adult.py
```

In case you want to run the full experiment, you can use the `run_experiment_seq.sh` script. **NOTE** this will take a very long while, since it runs all the repeats in sequence.

We will also include the slurm workload manager script, that was used to run the example (see `slurm_adult.sh`).
