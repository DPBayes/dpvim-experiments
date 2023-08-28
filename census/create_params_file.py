from itertools import product

#
nondp_params = False

if not nondp_params:
    epsilons = [1.]
    seeds = range(123, 123 + 10)
    variants = [('vanilla', 2.0), ('aligned', 2.0), ('precon', 4.0)]
    inits = [1.0]
    epochss = [200, 1000]
    file_path = "dp_run_params.txt"

    with open(file_path, "w") as f:
        for (seed, eps, (variant, C), init, epochs) in product(seeds, epsilons, variants, inits, epochss):
            f.writelines(f"{seed} {eps} {variant} {C} {init} {epochs}\n")

else:
    seeds = range(123, 123 + 50)
    epochss = [2000]
    file_path = "nondp_run_params.txt"
    with open(file_path, "w") as f:
        for (seed, epochs) in product(seeds, epochss):
            f.writelines(f"{seed} {epochs}\n")
