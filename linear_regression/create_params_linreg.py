from itertools import product

epsilons = [1.]
seeds = range(123, 123+50)
ndimss = [100, 200]
# corr_strengths = [0.1, 1000.] # with LKJ
# filename = "params_linreg_toy_lkj.txt"
corr_strengths = [0.2, 0.8] # with manual
filename = "params_linreg_toy_manual.txt"

with open(filename, "w") as f:
    for epsilon, seed, corr_strength, ndims in product(epsilons, seeds, corr_strengths, ndimss):
        f.write(f"{seed} {epsilon} {corr_strength} {ndims}\n")

