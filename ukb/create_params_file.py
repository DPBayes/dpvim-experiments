from itertools import product

# create the parameter file used both by the main_aligned.py and the main_vanilla.py
epsilons = [1., 2., 4., 10.]
seeds = range(123, 133)

with open("dp_run_params.txt", "w") as f:
    for (seed, eps) in product(seeds, epsilons):
        f.writelines(f"{seed} {eps}\n")
