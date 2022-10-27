"""
This script is used to plot Figure 1 and 2 (as well as various figures in the supplement).
"""

import argparse, os, sys, pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils import filenamer, neurips_fig_style

from collections import defaultdict

parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
parser.add_argument('data_path', type=str, help='Path to input data.')
parser.add_argument("stored_pickle_dir", type=str, help="Dir from which to read traces.")
parser.add_argument("nondp_pickle_dir", type=str, help="Dir from which to read nondp baseline traces")
parser.add_argument("--output_dir", type=str, default="./", help="Dir to store the results")
def parse_clipping_threshold_arg(value): return None if value == "None" else float(value)
parser.add_argument("--clipping_threshold", default=None, type=parse_clipping_threshold_arg, help="Clipping threshold for DP-SGD.")
parser.add_argument("--epsilon", default=1.0, type=float)
parser.add_argument("--k", default=16, type=int, help="Mixture components in fit.")
parser.add_argument("--num_epochs", "-e", default=1000, type=int, help="Number of training epochs.")
parser.add_argument("--init_scale", type=float, default=0.1)

parser.add_argument("--plot_traces", action='store_true', default=False)
parser.add_argument("--plot_robustness", action='store_true', default=False)
parser.add_argument("--plot_robustness_tradeoff", action='store_true', default=False)
parser.add_argument("--plot_error_over_epochs", action='store_true', default=False)
parser.add_argument("--plot_ng_variant_error_trace", action='store_true', default=False)

parser.add_argument("--clip_ng", type=float, default=0.1)
parser.add_argument("--error_type", choices=['stderr', 'quantiles'], default='stderr')

args, unknown_args = parser.parse_known_args()

init_scale = args.init_scale

##########################################
plt.rcParams.update(neurips_fig_style)

########################################## Read original data
# read the whole UKB data
df_whole = pd.read_csv(args.data_path)
print("Loaded data set has {} rows (entries) and {} columns (features).".format(*df_whole.shape))

train_df_whole = df_whole.copy()
train_df_whole = train_df_whole.dropna()

from twinify.model_loading import load_custom_numpyro_model

model, _, preprocess_fn, postprocess_fn = load_custom_numpyro_model("model.py", args, unknown_args, train_df_whole)

########################################## Set up guide

from smart_auto_guide import SmartAutoGuide

from numpyro.infer.autoguide import AutoDiagonalNormal

observation_sites = {'X', 'y'}
guide = SmartAutoGuide.wrap_for_sampling(AutoDiagonalNormal, observation_sites)(model)
guide.initialize()

##########################################

from utils import traces, init_proportional_abs_error

def load_traces_for_run(path, prefix, epsilon, seed, clipping_threshold, args):
    output_name = filenamer(prefix, "", args, epsilon=epsilon, seed=seed, clipping_threshold=clipping_threshold)
    output_path = f"{os.path.join(path, output_name)}_traces.p"

    trace = pd.read_pickle(output_path)
    loc_trace = guide._unpack_latent(trace.loc_trace)
    scale_trace = guide._unpack_latent(trace.scale_trace)

    poisson_regression_keys = ["covid_intercept"] + [key for key in loc_trace.keys() if "covid_weight" in key]
    poisson_regression_keys = list(sorted(poisson_regression_keys))

    # gather the traces responsible for covid weights and intercept
    regression_loc_trace = np.hstack([
                                        loc_trace[key] if "intercept" not in key
                                        else loc_trace[key][:, np.newaxis]
                                        for key in poisson_regression_keys
                                    ])
    regression_scale_trace = np.hstack([
                                        scale_trace[key] if "intercept" not in key
                                        else scale_trace[key][:, np.newaxis]
                                        for key in poisson_regression_keys
                                    ])
    return regression_loc_trace, regression_scale_trace

def load_traces_for_run_and_compute_error(path, prefix, epsilon, seed, clipping_threshold, args, nondp_regression_loc_trace, nondp_regression_scale_trace):
    regression_loc_trace, regression_scale_trace = load_traces_for_run(path, prefix, epsilon, seed, clipping_threshold, args)

    # compute the proportional absolute error against the non-DP model
    loc_errors = init_proportional_abs_error(regression_loc_trace, nondp_regression_loc_trace, False)
    scale_errors = init_proportional_abs_error(regression_scale_trace, nondp_regression_scale_trace, False)

    return loc_errors, scale_errors

#########################

### load non-DP parameter traces
og_epochs = args.num_epochs
args.num_epochs = 1000
seeds = list(range(123, 133))
nondp_regression_loc_trace, nondp_regression_scale_trace = [], []
for seed in seeds:

    loc_trace, scale_trace = load_traces_for_run(
        args.nondp_pickle_dir,
        "vanilla", "non_dp", seed=seed, clipping_threshold=None, args=args
    )
    nondp_regression_loc_trace.append(loc_trace)
    nondp_regression_scale_trace.append(scale_trace)

nondp_regression_loc_trace = np.mean(nondp_regression_loc_trace, axis=0)
nondp_regression_scale_trace = np.mean(nondp_regression_scale_trace, axis=0)
args.num_epochs = og_epochs

# note: averaging over these messes up trace in early iterations (where they are different not only because
#   of SVI randomness but because they have not converged yet from different inits),
#   but we don't care about that since we only ever look at the last iteration of the baseline anyways..
#########################

err_type_suffix = f'_{args.error_type}'
if args.error_type == 'stderr':
    def compute_yerr(plot_data, axis):
            return plot_data.std(ddof=1, axis=axis) / np.sqrt(plot_data.shape[axis])
else:
    quantiles = (.1, .9)
    def compute_yerr(plot_data, axis):
            return (np.quantile(plot_data, quantiles, axis=axis) - np.mean(plot_data, axis=axis)) * np.array([[-1], [1]])


PREFIXES = ['vanilla', 'aligned', 'precon'] #, 'ng', 'aligned_ng']
Cs = {'vanilla': 2.0, 'aligned':2.0, 'precon': 4.0, 'ng': args.clip_ng, 'aligned_ng': args.clip_ng}
if args.plot_traces:

    ### load DP results
    stored_pickle_dir = os.path.join(args.stored_pickle_dir, f"init{init_scale}/")

    seeds = list(range(123, 133))
    epsilons = [args.epsilon]

    trace_errors_loc = dict()
    trace_errors_scale = dict()

    for prefix in PREFIXES:

        trace_errors_prefix_loc = defaultdict(dict)
        trace_errors_prefix_scale = defaultdict(dict)

        for epsilon in epsilons:
            for seed in seeds:
                loc_errors, scale_errors = load_traces_for_run_and_compute_error(
                    stored_pickle_dir, prefix, epsilon, seed, Cs[prefix], args,
                    nondp_regression_loc_trace, nondp_regression_scale_trace
                )

                trace_errors_prefix_loc[epsilon][seed] = loc_errors
                trace_errors_prefix_scale[epsilon][seed] = scale_errors

        trace_errors_loc[prefix] = trace_errors_prefix_loc
        trace_errors_scale[prefix] = trace_errors_prefix_scale


    ### plot
    thinning = 10
    T = args.num_epochs
    ts = np.arange(0, T + 1)[::thinning] # +1 because first step is initialisation, then one after each epoch

    #### average over all the parameters
    fig, axis = plt.subplots(nrows=2, sharex=True)
    plot_name = filenamer("avg_trace_error", f"aligned_vs_vanilla_init{init_scale}_precon_C4", args, seed="all") + err_type_suffix

    for prefix in PREFIXES:
        ## locs
        loc_plot_data = np.array(list(trace_errors_loc[prefix][args.epsilon].values())).mean(-1)[:, ::thinning]
        axis[0].errorbar(
                ts,
                loc_plot_data.mean(0),
                compute_yerr(loc_plot_data, axis=0),
                alpha=0.2, label=prefix
        )

        ## scales
        scale_plot_data = np.array(list(trace_errors_scale[prefix][args.epsilon].values())).mean(-1)[:, ::thinning]
        axis[1].errorbar(
                ts,
                scale_plot_data.mean(0),
                compute_yerr(scale_plot_data, axis=0),
                alpha=0.2
        )

    axis[1].set_xlabel("current epoch")
    axis[0].set_ylabel("MPAE for mean")
    axis[1].set_ylabel("MPAE for std")
    handles, labels = axis[0].get_legend_handles_labels()
    axis[1].legend(handles, labels)


    fig.suptitle(f'Evolution of error in learned parameters\n$\\varepsilon=${args.epsilon}', fontsize=11.5)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{plot_name}.pdf", format="pdf", bbox_inches='tight')

    axis[0].set_yscale('log')
    axis[1].set_yscale('log')
    plt.savefig(f"{args.output_dir}/{plot_name}_logscale.pdf", format="pdf", bbox_inches='tight')

    plt.close()

if args.plot_robustness:
    plt.rcParams.update({"font.size":12})
    trace_errors_loc = defaultdict(lambda: defaultdict(dict))
    trace_errors_scale = defaultdict(lambda: defaultdict(dict))

    seeds = list(range(123, 133))
    init_scales = [0.01, 0.032, 0.1, 0.316, 1.0]

    for init_scale in init_scales:
        ### load DP results
        stored_pickle_dir = os.path.join(args.stored_pickle_dir, f"init{init_scale}/")

        for prefix in PREFIXES:
            for seed in seeds:
                loc_errors, scale_errors = load_traces_for_run_and_compute_error(
                    stored_pickle_dir, prefix, args.epsilon, seed, Cs[prefix], args,
                    nondp_regression_loc_trace, nondp_regression_scale_trace
                )

                trace_errors_loc[prefix][init_scale][seed] = loc_errors
                trace_errors_scale[prefix][init_scale][seed] = scale_errors

    #### average over all the parameters
    fig, axis = plt.subplots(nrows=2, sharex=True, sharey=False)
    plot_name = filenamer("avg_final_error_for_init", f"aligned_vs_vanilla_init{init_scale}_precon_C4.0", args, seed="all") + err_type_suffix

    compare_at_iteration = -1

    for prefix in PREFIXES:
        loc_plot_data = np.array([np.array(list(seeds_dict.values())) for seeds_dict in trace_errors_loc[prefix].values()]).mean(axis=-1)[:,:,compare_at_iteration]
        axis[0].errorbar(
            init_scales,
            loc_plot_data.mean(axis=-1),
            compute_yerr(loc_plot_data, axis=-1),
            label=prefix,
            alpha=.7
        )

        scale_plot_data = np.array([np.array(list(seeds_dict.values())) for seeds_dict in trace_errors_scale[prefix].values()]).mean(axis=-1)[:,:,compare_at_iteration]
        axis[1].errorbar(
            init_scales,
            scale_plot_data.mean(axis=-1),
            compute_yerr(scale_plot_data, axis=-1),
            label=prefix,
            alpha=.7
        )

    axis[1].set_xticks(init_scales)
    axis[1].set_xlabel("initial values for $\sigma_q$")
    axis[0].set_ylabel("MPAE in mean")
    axis[1].set_ylabel("MPAE in std.")

    #axis[0].legend()
    axis[1].legend()

    fig.suptitle("Error in parameters over initialization\n" + f'$\epsilon={args.epsilon}$, after {args.num_epochs} epochs', fontsize=11.5)

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    plt.savefig(f"{args.output_dir}/{plot_name}.pdf", format="pdf", bbox_inches='tight')

    axis[0].set_yscale('log')
    axis[1].set_yscale('log')
    axis[1].set_xscale('log')
    plt.savefig(f"{args.output_dir}/{plot_name}_logscale.pdf", format="pdf", bbox_inches='tight')

    plt.close()

if args.plot_robustness_tradeoff:
    ## plots the errors from initializion to 2dim error plot
    plt.rcParams.update({"font.size":12})
    trace_errors_loc = defaultdict(lambda: defaultdict(dict))
    trace_errors_scale = defaultdict(lambda: defaultdict(dict))

    seeds = list(range(123, 133))
    init_scales = [0.01, 0.032, 0.1, 0.316, 1.0]
    for init_scale in init_scales:
        ### load DP results
        stored_pickle_dir = os.path.join(args.stored_pickle_dir, f"init{init_scale}/")

        for prefix in PREFIXES:
            for seed in seeds:
                loc_errors, scale_errors = load_traces_for_run_and_compute_error(
                    stored_pickle_dir, prefix, args.epsilon, seed, Cs[prefix], args,
                    nondp_regression_loc_trace, nondp_regression_scale_trace
                )

                trace_errors_loc[prefix][init_scale][seed] = loc_errors
                trace_errors_scale[prefix][init_scale][seed] = scale_errors

    #### average over all the parameters
    compare_at_iteration = -1

    fig, axis = plt.subplots()
    plot_name = filenamer("error_tradeoff_for_init", f"aligned_vs_vanilla_init{init_scale}_precon_C4.0", args, seed="all") + err_type_suffix

    for prefix in PREFIXES:
        loc_plot_data = np.array([np.array(list(seeds_dict.values())) for seeds_dict in trace_errors_loc[prefix].values()]).mean(axis=-1)[:,:,compare_at_iteration]
        scale_plot_data = np.array([np.array(list(seeds_dict.values())) for seeds_dict in trace_errors_scale[prefix].values()]).mean(axis=-1)[:,:,compare_at_iteration]

        data_to_plot = np.array(sorted(zip(scale_plot_data.mean(axis=-1), loc_plot_data.mean(axis=-1)))).T
        axis.plot(
            *data_to_plot,
            label=prefix,
            alpha=.7
        )

    axis.set_xlabel("MPAE in std.")
    axis.set_ylabel("MPAE in mean")

    axis.legend()

    fig.suptitle(f"Accuracy trade-off between the variational parameters\n $\epsilon=${args.epsilon}, after {args.num_epochs} epochs", fontsize=11.5)
    fig.tight_layout()

    plt.savefig(f"{args.output_dir}/{plot_name}.pdf", format="pdf", bbox_inches='tight')

    axis.set_xscale('log')
    axis.set_yscale('log')

    fig.tight_layout()
    plt.subplots_adjust(left=0.20)
    plt.savefig(f"{args.output_dir}/{plot_name}_logscale.pdf", format="pdf", bbox_inches='tight')

    plt.close()

#### Plot error over different num_epochs leading to fixed epsilon
if args.plot_error_over_epochs:
    plt.rcParams.update({"font.size":12})
    trace_errors_loc = defaultdict(lambda: defaultdict(dict))
    trace_errors_scale = defaultdict(lambda: defaultdict(dict))

    seeds = list(range(123, 133))
    n_epochs = [200, 400, 600, 800, 1000, 2000, 4000, 8000]
    C_from_args = args.clipping_threshold
    T_from_args = args.num_epochs
    for n_epoch in n_epochs:
        args.num_epochs = n_epoch
        ### load DP results
        stored_pickle_dir = os.path.join(args.stored_pickle_dir, f"init{args.init_scale}/")

        for prefix in PREFIXES:
            for seed in seeds:
                loc_errors, scale_errors = load_traces_for_run_and_compute_error(
                    stored_pickle_dir, prefix, args.epsilon, seed, Cs[prefix], args,
                    nondp_regression_loc_trace, nondp_regression_scale_trace
                )

                trace_errors_loc[prefix][n_epoch][seed] = loc_errors
                trace_errors_scale[prefix][n_epoch][seed] = scale_errors

    # revert T back from what it was in the arguments
    args.num_epochs = T_from_args

    #### average over all the parameters
    fig, axis = plt.subplots(nrows=2, sharex=True)
    plot_name = f"avg_final_error_for_num_epochs_init{args.init_scale}_eps{args.epsilon}_maxT{n_epochs[-1]}_{args.error_type}"

    compare_at_iteration = -1

    for prefix in PREFIXES:
        loc_plot_data = np.array([np.array(list(seeds_dict.values()))[:,-1,:] for seeds_dict in trace_errors_loc[prefix].values()]).mean(axis=-1)
        axis[0].errorbar(
            n_epochs,
            loc_plot_data.mean(axis=-1),
            compute_yerr(loc_plot_data, axis=-1),
            label=prefix,
            alpha=.7
        )

        scale_plot_data = np.array([np.array(list(seeds_dict.values()))[:,-1,:] for seeds_dict in trace_errors_scale[prefix].values()]).mean(axis=-1)
        axis[1].errorbar(
            n_epochs,
            scale_plot_data.mean(axis=-1),
            compute_yerr(scale_plot_data, axis=-1),
            label=prefix,
            alpha=.7
        )


    axis[1].set_xlabel("Total number of training epochs")
    axis[0].set_ylabel("MPAE in mean")
    axis[1].set_ylabel("MPAE in std")

    axis[0].legend()

    fig.suptitle(f'Error in parameters after different training lengths\n$\\varepsilon=${args.epsilon}', fontsize=11.5)

    plt.savefig(f"{args.output_dir}/{plot_name}.pdf", format="pdf", bbox_inches='tight')


    axis[0].set_yscale('log')
    axis[1].set_yscale('log')
    plt.savefig(f"{args.output_dir}/{plot_name}_logscale.pdf", format="pdf", bbox_inches='tight')

    plt.close()

if args.plot_ng_variant_error_trace:
    og_PREFIXES = PREFIXES
    PREFIXES = ['vanilla', 'aligned', 'ng', 'aligned_ng']
    epsilon = args.epsilon

    stored_pickle_dir = os.path.join(args.stored_pickle_dir, f"init{init_scale}/")

    seeds = list(range(123, 133))

    trace_errors_loc = dict()
    trace_errors_scale = dict()

    for prefix in PREFIXES:

        trace_errors_prefix_loc = defaultdict(dict)
        trace_errors_prefix_scale = defaultdict(dict)

        for seed in seeds:
            loc_errors, scale_errors = load_traces_for_run_and_compute_error(
                stored_pickle_dir, prefix, epsilon, seed, Cs[prefix], args,
                nondp_regression_loc_trace, nondp_regression_scale_trace
            )

            trace_errors_prefix_loc[epsilon][seed] = loc_errors
            trace_errors_prefix_scale[epsilon][seed] = scale_errors

        trace_errors_loc[prefix] = trace_errors_prefix_loc
        trace_errors_scale[prefix] = trace_errors_prefix_scale


    ### plot
    fig, axis = plt.subplots(figsize=(6, 4), nrows=2, sharex=True)

    thinning = 10
    T = args.num_epochs
    ts = np.arange(0, T + 1 + 1)[::thinning]

    for prefix in PREFIXES:
        ## locs
        loc_plot_data = np.array(list(trace_errors_loc[prefix][args.epsilon].values())).mean(-1)[:, ::thinning]
        axis[0].errorbar(
                ts,
                loc_plot_data.mean(0),
                compute_yerr(loc_plot_data, axis=0),
                alpha=0.2, label=prefix
        )

        ## scales
        scale_plot_data = np.array(list(trace_errors_scale[prefix][args.epsilon].values())).mean(-1)[:, ::thinning]
        axis[1].errorbar(
                ts,
                scale_plot_data.mean(0),
                compute_yerr(scale_plot_data, axis=0),
                alpha=0.2
        )

    axis[1].set_xlabel("current epoch")
    axis[0].set_ylabel("MPAE for mean")
    axis[1].set_ylabel("MPAE for std")
    handles, labels = axis[0].get_legend_handles_labels()

    new_labels = []
    for label in labels:
        if label == "ng":
            new_labels.append("Natural grad.")
        elif label == "aligned_ng":
            new_labels.append("Aligned natural grad")
        elif label == "aligned":
            new_labels.append("Aligned grad.")
        elif label == "vanilla":
            new_labels.append("Vanilla")
        else:
            new_labels.append(label)

    #axis[1].legend(handles, new_labels)
    fig.legend(handles, new_labels, fontsize=7)


    fig.suptitle(f'Evolution of error in learned parameters\n $\\varepsilon=${args.epsilon}, naturalgradient clipping threshold: {args.clip_ng}', fontsize=11.5)

    plt.tight_layout()

    axis[0].set_yscale('log')
    axis[1].set_yscale('log')
    plot_name = f"avg_trace_error_{args.epsilon}_k16_seedall_epochs{args.num_epochs}_ng_comparison_init{args.init_scale}_stderr_logscale_ngC{args.clip_ng}.pdf"
    plt.savefig(f"{args.output_dir}/{plot_name}", format="pdf", bbox_inches='tight')
    plt.close()

    PREFIXES = og_PREFIXES
