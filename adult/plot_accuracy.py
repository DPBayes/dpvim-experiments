import jax, pickle, argparse, os

import pandas as pd
import numpy as np

import jax.numpy as jnp

from numpyro.infer.autoguide import AutoDiagonalNormal

from utils import infer

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


N_train = len(X_train)
init_auto_scale = args.init_auto_scale

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
seeds = range(123,123 + 50)
epsilons = [1.0, 2.0, 4.0, 10.]
init_auto_scales = [1.0]


ng_dpsvi_l2s = np.zeros([len(epsilons), len(seeds), len(init_auto_scales), 2])
vanilla_dpsvi_l2s = np.zeros([len(epsilons), len(seeds), len(init_auto_scales), 2])
aligned_dpsvi_l2s = np.zeros([len(epsilons), len(seeds), len(init_auto_scales), 2])
aligned_ng_dpsvi_l2s = np.zeros([len(epsilons), len(seeds), len(init_auto_scales), 2])
precon_dpsvi_l2s = np.zeros([len(epsilons), len(seeds), len(init_auto_scales), 2])

def compute_metrics_and_update_arrays(variant, args, l2_array, indx):
    iter_eps, i, j = indx
    try:
        trace = pd.read_pickle(filenamer(variant, "trace", Cs[variant]) + ".p")
        params_last_epoch = trace[-1]

        l2_array[iter_eps,i,j,0] = np.linalg.norm(params_last_epoch["auto_loc"]-nondp_svi_params["auto_loc"])
        l2_array[iter_eps,i,j,1] = np.linalg.norm(params_last_epoch["auto_scale"]-nondp_svi_params["auto_scale"])

    except:
        filename = filenamer(variant, "trace", Cs[variant]) + ".p"
        print(f"Didn't find {filename}")
        l2_array[iter_eps,i,j] = np.nan * np.empty_like(l2_array[iter_eps,i,j])


for iter_eps, eps in enumerate(epsilons):
    args.epsilon = eps
    print(f"Going over epsilon {eps}")
    ## compute results over seeds
    # NG
    for i, seed in enumerate(seeds):
        args.seed = seed
        for j, init_auto_scale in enumerate(init_auto_scales):
            args.init_auto_scale = init_auto_scale
            compute_metrics_and_update_arrays("ng", args,
                    ng_dpsvi_l2s,
                    (iter_eps, i, j)
                    )

    # Vanilla
    for i, seed in enumerate(seeds):
        args.seed = seed
        for j, init_auto_scale in enumerate(init_auto_scales):
            args.init_auto_scale = init_auto_scale
            compute_metrics_and_update_arrays("vanilla", args,
                    vanilla_dpsvi_l2s,
                    (iter_eps, i, j)
                    )

    # Aligned
    for i, seed in enumerate(seeds):
        args.seed = seed
        for j, init_auto_scale in enumerate(init_auto_scales):
            args.init_auto_scale = init_auto_scale
            compute_metrics_and_update_arrays("aligned", args,
                    aligned_dpsvi_l2s,
                    (iter_eps, i, j)
                    )

    # Aligned NG
    for i, seed in enumerate(seeds):
        args.seed = seed
        for j, init_auto_scale in enumerate(init_auto_scales):
            args.init_auto_scale = init_auto_scale
            compute_metrics_and_update_arrays("aligned_ng", args,
                    aligned_ng_dpsvi_l2s,
                    (iter_eps, i, j)
                    )

    # Precon
    for i, seed in enumerate(seeds):
        args.seed = seed
        for j, init_auto_scale in enumerate(init_auto_scales):
            args.init_auto_scale = init_auto_scale
            args.init_auto_scale = init_auto_scale
            compute_metrics_and_update_arrays("precon", args,
                    precon_dpsvi_l2s,
                    (iter_eps, i, j)
                    )

#### Plot
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size":9})
### l2 error

## plot
n_seeds = len(seeds)

def plot_metric(vanilla_plot_data, ng_plot_data, aligned_plot_data, aligned_ng_plot_data, precon_plot_data, axis):
    axis.errorbar(
             epsilons,
             np.nanmean(vanilla_plot_data, axis=1),
             yerr=np.nanstd(vanilla_plot_data, axis=1)/np.sqrt(n_seeds),
             color="red", label="vanilla", alpha=0.5
             )
    axis.errorbar(
             epsilons,
             np.nanmean(ng_plot_data, axis=1),
             yerr=np.nanstd(ng_plot_data, axis=1)/np.sqrt(n_seeds),
             color="blue", label="natural grad.", alpha=0.5
             )
    axis.errorbar(
             epsilons,
             np.nanmean(aligned_plot_data, axis=1),
             yerr=np.nanstd(aligned_plot_data, axis=1)/np.sqrt(n_seeds),
             color="teal", label="aligned", alpha=0.5
             )
    axis.errorbar(
             epsilons,
             np.nanmean(aligned_ng_plot_data, axis=1),
             yerr=np.nanstd(aligned_plot_data, axis=1)/np.sqrt(n_seeds),
             color="cyan", label="aligned NG", alpha=0.5
             )
    axis.errorbar(
             epsilons,
             np.nanmean(precon_plot_data, axis=1),
             yerr=np.nanstd(precon_plot_data, axis=1)/np.sqrt(n_seeds),
             color="magenta", label="precon", alpha=0.5
             )


## Plot l2-error
for auto_scale_iter, init_auto_scale in enumerate(init_auto_scales):
    fig, axis = plt.subplots(figsize=(5., 2.5), ncols=2)
    plot_metric(
            vanilla_dpsvi_l2s[:,:,auto_scale_iter,0],
            ng_dpsvi_l2s[:,:,auto_scale_iter,0],
            aligned_dpsvi_l2s[:,:,auto_scale_iter,0],
            aligned_ng_dpsvi_l2s[:,:,auto_scale_iter,0],
            precon_dpsvi_l2s[:,:,auto_scale_iter,0],
            axis[0]
            )
    plot_metric(
            vanilla_dpsvi_l2s[:,:,auto_scale_iter,1],
            ng_dpsvi_l2s[:,:,auto_scale_iter,1],
            aligned_dpsvi_l2s[:,:,auto_scale_iter,1],
            aligned_ng_dpsvi_l2s[:,:,auto_scale_iter,1],
            precon_dpsvi_l2s[:,:,auto_scale_iter,1],
            axis[1]
            )
    axis[0].set_ylim(0.0)
    axis[0].set_xlabel("$\epsilon$")
    axis[1].set_xlabel("$\epsilon$")
    axis[0].set_ylabel("l2-error in means")
    axis[1].set_ylabel("l2-error in stds")
    axis[1].set_ylim(0.0)

    fig.legend(*(axis[0].get_legend_handles_labels()), loc="center", bbox_to_anchor=[0.5, 0.0], ncol=5, fontsize=7)
    plt.subplots_adjust(wspace=0.25, top=0.85, bottom=0.2)
    plt.savefig(f"{args.figure_path}/l2_adam_init_scale{init_auto_scale}.pdf", format="pdf", bbox_inches="tight")
    plt.close()

