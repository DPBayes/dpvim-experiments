from itertools import product

epsilons = [1., 2., 4.0, 10.]
seeds = range(123, 123+50)

with open("params_adult.txt", "w") as f:
    for epsilon, seed in product(epsilons, seeds):
        f.write(f"{epsilon} {seed}\n")

