import jax, pickle, argparse, os

import pandas as pd
import numpy as np

import jax.numpy as jnp

from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.distributions.transforms import SoftplusTransform


parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
parser.add_argument("--num_epochs", "-e", default=4000, type=int, help="Number of training epochs.")
parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="Subsampling ratio for DP-SGD.")
parser.add_argument("--results_path", type=str, default="./results/", help="Path to folder in which parameter traces are stored'")
parser.add_argument("--figure_path", type=str, default="./figures")
parser.add_argument("--init_auto_scale", type=float, default=1.0, help="Initial std for the VI posterior")
parser.add_argument("--optim", type=str, default="adam", help="Gradient optimizer")
parser.add_argument("--sgd_lr", type=float, default=1e-6, help="Learning rate for the SGD optimizer")
args, unknown_args = parser.parse_known_args()

## load data
from main_adult import model, add_intercept
from load_adult import preprocess_adult, load_orig_data

preprocessed_train_data, preprocessed_test_data, encodings, data_description = preprocess_adult()

X_train = preprocessed_train_data.copy()
y_train = X_train.pop("income")

X_test = preprocessed_test_data.copy()
y_test = X_test.pop("income")

## adjust the categorical variables with the largest group
for key, value in data_description.items():
    if value != "continuous" and key != "income":
        categorical_names = [name for name in preprocessed_train_data.columns if key in name]
        largest_category_indx = preprocessed_train_data[categorical_names].sum().argmax()
        largest_category = categorical_names[largest_category_indx]
        del X_train[largest_category]
        del X_test[largest_category]


## add intercept
X_train = add_intercept(X_train.values)
y_train = jnp.array(y_train.values)

X_test = add_intercept(X_test.values)
y_test = y_test.values

guide = AutoDiagonalNormal(model, init_scale=args.init_auto_scale)

num_epochs = args.num_epochs
q = args.sampling_ratio

N_train = len(X_train)
init_auto_scale = args.init_auto_scale

C_ng = 0.1 # chosen by hand
C_vanilla = 3.0 # chosen by hand
C_aligned = 3.0 # chosen by hand
C_aligned_ng = 0.1 # chosen by hand
C_precon = 4.0 # chosen by hand

Cs = {
    'ng': C_ng,
    'vanilla': C_vanilla,
    'aligned': C_aligned,
    'aligned_ng': C_aligned_ng,
    'precon': C_precon
}

######
## filename template
def filenamer(variant, suffix, C):
    if args.optim == "adam":
        return f"{args.results_path}/adult_{variant}_ne{num_epochs}_C{C}_q{q}_eps{args.epsilon}_auto_scale{args.init_auto_scale}_seed{seed}_optim{args.optim}_{suffix}"
    elif args.optim == "sgd":
        return f"{args.results_path}/adult_{variant}_ne{num_epochs}_C{C}_q{q}_eps{args.epsilon}_auto_scale{args.init_auto_scale}_seed{seed}_optim{args.optim}_lr{args.sgd_lr}_{suffix}"
    ##
##

# fit non-dp svi
nondp_filename = f"{args.results_path}/adult_nondp_params_auto_scale0.1_T100000_seed123.p"
if not os.path.exists(nondp_filename):
    from numpyro.infer.svi import SVI
    from numpyro.infer import Trace_ELBO
    from numpyro.optim import Adam
    optimiser = Adam(1e-3)
    svi = SVI(model, guide, optimiser, Trace_ELBO(), N=N_train)
    nondp_svi = svi.run(jax.random.PRNGKey(123), int(1e5), X_train, y_train)
    nondp_svi_params = nondp_svi.params
    pd.to_pickle(nondp_svi_params, nondp_filename)
else:
    nondp_svi_params = pd.read_pickle(nondp_filename)

####
# Compute the noise std from the trace and compare to std amongst last iterands
from utils import single_param_site_linear_fit
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size":9})

variants = ["aligned", "vanilla"]
seeds = np.arange(123, 123+50)
epsilons = [1.0, 2.0, 4.0, 10.0]

if args.optim == "adam":
    optim_fname_part = "optimadam"
else:
    optim_fname_part = f"optimsgd_{args.sgd_lr}"


#### linear regression heuristic for the convergence check

for variant in variants:
    loc_stds = np.zeros([len(epsilons), len(seeds), X_train.shape[1]])
    scale_stds = np.zeros([len(epsilons), len(seeds), X_train.shape[1]])

    loc_indices = np.zeros([len(epsilons), len(seeds), X_train.shape[1]])
    scale_indices = np.zeros([len(epsilons), len(seeds), X_train.shape[1]])

    loc_last_iterands = np.zeros([len(epsilons), len(seeds), X_train.shape[1]])
    scale_last_iterands  = np.zeros([len(epsilons), len(seeds), X_train.shape[1]])

    for eps_iter, epsilon in enumerate(epsilons):
        args.epsilon = epsilon
        for seed_iter, seed in enumerate(seeds):
            args.seed = seed
            trace = pd.read_pickle(filenamer(variant, "trace", Cs[variant]) + ".p")
            trace_dict = {key: np.vstack([elem[key] for elem in trace]) for key in trace[0].keys()}
            trace_dict["auto_scale"] = SoftplusTransform().inv(trace_dict["auto_scale"])

            linreg_coefs_loc, _ = single_param_site_linear_fit(trace_dict["auto_loc"], spacing=400)
            linreg_coefs_scale, _ = single_param_site_linear_fit(trace_dict["auto_scale"], spacing=400)

            ### assess which are converged
            ts = np.arange(0, args.num_epochs, 400)
            #loc
            converged_after_epoch_loc = []
            for j in range(X_train.shape[1]):
                indices = np.where(np.abs(linreg_coefs_loc[:,j])<0.05)[0]
                if len(indices) > 0:
                    converged_after_epoch_loc.append(ts[indices[0]])
                else:
                    converged_after_epoch_loc.append(-1)
            converged_after_epoch_loc = np.array(converged_after_epoch_loc)

            #scale
            converged_after_epoch_scale = []
            for j in range(X_train.shape[1]):
                indices = np.where(np.abs(linreg_coefs_scale[:,j])<0.05)[0]
                if len(indices) > 0:
                    converged_after_epoch_scale.append(ts[indices[0]])
                else:
                    converged_after_epoch_scale.append(-1)
            converged_after_epoch_scale = np.array(converged_after_epoch_scale)

            # compute the std from trace
            for j in range(X_train.shape[1]):
                cnvrg_indx = converged_after_epoch_loc[j]
                if cnvrg_indx != -1:
                    loc_stds[eps_iter, seed_iter, j] = trace_dict["auto_loc"][cnvrg_indx:, j].std()
                else:
                    loc_stds[eps_iter, seed_iter, j] = np.nan
                loc_indices[eps_iter, seed_iter, j] = cnvrg_indx

            for j in range(X_train.shape[1]):
                cnvrg_indx = converged_after_epoch_scale[j]
                if cnvrg_indx != -1:
                    scale_stds[eps_iter, seed_iter, j] = trace_dict["auto_scale"][cnvrg_indx:, j].std()
                else:
                    scale_stds[eps_iter, seed_iter, j] = np.nan
                scale_indices[eps_iter, seed_iter, j] = cnvrg_indx


            # store the last iterands
            loc_last_iterands[eps_iter, seed_iter] = trace_dict["auto_loc"][-1]
            scale_last_iterands[eps_iter, seed_iter] = trace_dict["auto_scale"][-1]

    # compute the errors
    loc_mses = np.zeros([len(epsilons), len(seeds)])
    scale_mses = np.zeros([len(epsilons), len(seeds)])

    loc_mses_baseline = np.zeros([len(epsilons), len(seeds)])
    scale_mses_baseline = np.zeros([len(epsilons), len(seeds)])

    for eps_iter, epsilon in enumerate(epsilons):
        loc_last_iterand_std = np.zeros(X_train.shape[1])
        scale_last_iterand_std = np.zeros(X_train.shape[1])
        for j in range(X_train.shape[1]):
            loc_last_iterand_std[j] = loc_last_iterands[eps_iter, :, j][np.where(loc_indices[eps_iter,:,j]!=-1)].std(0)
            scale_last_iterand_std[j] = scale_last_iterands[eps_iter, :, j][np.where(scale_indices[eps_iter,:,j]!=-1)].std(0)
        for seed_iter, seed in enumerate(seeds):
            loc_indx = np.where(loc_indices[eps_iter, seed_iter] != -1)[0]
            loc_std = loc_stds[eps_iter, seed_iter, loc_indx]
            loc_mses[eps_iter, seed_iter] = \
                    np.linalg.norm(loc_std - loc_last_iterand_std[loc_indx])**2 / len(loc_indx)
            loc_mses_baseline[eps_iter, seed_iter] = \
                    np.linalg.norm(loc_last_iterand_std[loc_indx])**2 / len(loc_indx)

            scale_indx = np.where(scale_indices[eps_iter, seed_iter] != -1)[0]
            scale_std = scale_stds[eps_iter, seed_iter, scale_indx]
            scale_mses[eps_iter, seed_iter] = \
                    np.linalg.norm(scale_std - scale_last_iterand_std[scale_indx])**2 / len(scale_indx)
            scale_mses_baseline[eps_iter, seed_iter] = \
                    np.linalg.norm(scale_last_iterand_std[scale_indx])**2 / len(scale_indx)


    fig, axis = plt.subplots(figsize=(4,2.5))

    loc_line = axis.errorbar(epsilons, loc_mses.mean(1), yerr=loc_mses.std(1), label="$q_m$")
    scale_line = axis.errorbar(epsilons, scale_mses.mean(1), yerr=scale_mses.std(1), label="$q_s$")

    axis.errorbar(epsilons, loc_mses_baseline.mean(1), yerr=loc_mses_baseline.std(1), label="$q_m$-baseline",
                    color=loc_line.lines[0]._color, ls="--")
    axis.errorbar(epsilons, scale_mses_baseline.mean(1), yerr=scale_mses_baseline.std(1), label="$q_s$-baseline",
                    color=scale_line.lines[0]._color, ls="--")

    axis.set_xlabel("$\epsilon$")

    axis.set_ylabel("MSE")
    axis.legend()
    plt.subplots_adjust(top=0.8)
    plt.savefig(f"{args.figure_path}/std_mseerror_linreg_{variant}_{optim_fname_part}_init_scale{init_auto_scale}.pdf", format="pdf", bbox_inches="tight")
    plt.close()
