This folder contains the code used to produce the Census experiment.

The data set for this experiment can be downloaded from [UCI machine learning repository](https://archive.ics.uci.edu/dataset/116/us+census+data+1990). After downloading the data, you need to unzip it. The data file used in the experiments is called `USCensus1990.data.txt`.

Similar to our other experiments, the code was originally ran in a cluster, paralellizing over the random seeds and epsilons. To generate the list of random seeds and epsilons, run
```
python create_params_file.py
```

We will also include the slurm workload manager script, that was used to run the example (see `slurm_infer.sh` and `slurm_infer_nondp.sh`).
