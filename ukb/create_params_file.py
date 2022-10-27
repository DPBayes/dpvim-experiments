from itertools import product

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("figure_num", type=int, help="Which figure to generate parameters for")
args = parser.parse_args()

if args.figure_num == 1:
    # parameter set experiment 1
    epsilons = [1.]
    seeds = range(123, 123 + 10)
    variants = [('vanilla', 2.0), ('aligned', 2.0), ('precon', 4.0)] # [(variant, clipping_threshold), ...]
    inits = [0.01, 0.032, 0.1, 0.316, 1.0]
    epochss = [1000]
    file_path = "dp_run_params_fig1.txt"
elif args.figure_num == 2:
    # parameter set experiment 2
    epsilons = [1.]
    seeds = range(123, 123 + 10)
    variants = [('vanilla', 2.0), ('aligned', 2.0), ('precon', 4.0)]
    inits = [1.0]
    epochss = [200, 400, 600, 800, 1000, 2000, 4000, 8000]
    file_path = "dp_run_params_fig2.txt"
elif args.figure_num == 3:
    # parameter set experiment 3
    epsilons = [1., 2., 4., 10.]
    seeds = range(123, 123 + 10)
    variants = [('aligned', 2.0)]
    inits = [1.0]
    epochss = [4000]
    file_path = "dp_run_params_fig3.txt"
else:
    raise ValueError(f"no run parameter set for figure number {args.figure_num}")

with open(file_path, "w") as f:
    for (seed, eps, (variant, C), init, epochs) in product(seeds, epsilons, variants, inits, epochss):
        f.writelines(f"{seed} {eps} {variant} {C} {init} {epochs}\n")
