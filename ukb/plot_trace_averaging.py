import argparse, os, sys, pickle
from re import L

import pandas as pd
import numpy as np

from numpyro.distributions.transforms import IdentityTransform, SoftplusTransform, biject_to
import matplotlib.pyplot as plt

from utils import filenamer, fit_model1, make_names_pretty, multirun_Rhat, split_Rhat, neurips_fig_style

from collections import defaultdict, namedtuple

from pvalue_utils import pool_analysis

parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
parser.add_argument('data_path', type=str, help='Path to input data.')
parser.add_argument("stored_pickle_dir", type=str, help="Dir from which to read traces.")
parser.add_argument("nondp_pickle_dir", type=str, help="Dir from which to read nondp baseline traces")
parser.add_argument("--output_dir", type=str, default="./", help="Dir to store the results")
parser.add_argument("--prefix", type=str, help="Type of a DPSVI")
def parse_clipping_threshold_arg(value): return None if value == "None" else float(value)
parser.add_argument("--clipping_threshold", default=None, type=parse_clipping_threshold_arg, help="Clipping threshold for DP-SGD.")
# parser.add_argument("--seed", default=None, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
# parser.add_argument("--epsilon", default=1.0, type=float)
parser.add_argument("--k", default=16, type=int, help="Mixture components in fit (for automatic modelling only).")
parser.add_argument("--num_epochs", "-e", default=1000, type=int, help="Number of training epochs.")
parser.add_argument("--init_scale", type=float, default=0.1)
parser.add_argument("--plot_traces", action='store_true', default=False)
parser.add_argument("--error_type", choices=['stderr', 'quantiles'], default='stderr')


args, unknown_args = parser.parse_known_args()

init_scale = args.init_scale

########################################## Read original data
# read the whole UKB data
df_whole = pd.read_csv(args.data_path)
print("Loaded data set has {} rows (entries) and {} columns (features).".format(*df_whole.shape))

train_df_whole = df_whole.copy()
train_df_whole = train_df_whole.dropna()

from twinify.model_loading import load_custom_numpyro_model

model, _, preprocess_fn, postprocess_fn = load_custom_numpyro_model("../twinify_models/model1_wholepop.py", args, unknown_args, train_df_whole)

train_data_whole, num_data_whole, feature_names = preprocess_fn(train_df_whole)

conditioned_postprocess_fn = lambda posterior_samples: postprocess_fn(
    posterior_samples, df_whole, feature_names
)
########################################## Set up guide

from smart_auto_guide import SmartAutoGuide

from numpyro.infer.autoguide import AutoDiagonalNormal

observation_sites = {'X', 'y'}
guide = SmartAutoGuide.wrap_for_sampling(AutoDiagonalNormal, observation_sites)(model)
guide.initialize()

##########################################

from utils import traces

def load_traces_for_run(path, prefix, epsilon, seed, clipping_treshold, args):
    wholepop_output_name = filenamer(prefix, args, epsilon=epsilon, seed=seed, clipping_threshold=clipping_treshold)
    wholepop_output_path = f"{os.path.join(path, wholepop_output_name)}_traces.p"

    trace = pd.read_pickle(wholepop_output_path)
    loc_trace = guide._unpack_latent(trace.loc_trace)
    scale_trace = guide._unpack_latent(trace.scale_trace)

    poisson_regression_keys = ['covid_weight_ethnicity']
    poisson_regression_keys = [key for key in loc_trace.keys() if "covid_weight" in key]
    if "covid_intercept" in loc_trace.keys():
        poisson_regression_keys = ["covid_intercept"] + poisson_regression_keys

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
    return traces(regression_loc_trace, regression_scale_trace)

from utils import average_params, tree_reduce, traces_to_dict
import jax

#########################

seeds = list(range(123, 133))
nondp_traces = []
### load non-DP parameter traces
for seed in seeds:
    nondp_regression_traces = load_traces_for_run(
        args.nondp_pickle_dir, args.prefix, "non_dp", seed, None, args
    )
    nondp_traces.append(traces_to_dict(nondp_regression_traces))

import jax
baseline_params = jax.tree_multimap(lambda *v: np.mean(np.array(v)[:,-1], axis=0), *nondp_traces)
num_params = baseline_params['auto_loc'].shape[-1]
assert baseline_params['auto_scale'].shape[-1] == num_params

# multirun_Rhat(nondp_traces, only_last=200, runs_are_first=True)
# assert False

# nondp_regression_traces = load_traces_for_run(
#     args.nondp_pickle_dir, args.prefix, "non_dp", 123, None, args
# )
# nondp_traces = traces_to_dict(nondp_regression_traces)
# baseline_params = jax.tree_map(lambda site_trace: site_trace[-1], nondp_traces)

# num_params = baseline_params['auto_loc'].shape[-1]
# assert baseline_params['auto_scale'].shape[-1] == num_params

#########################


### load DP results
seeds = list(range(123, 133))
epsilons = [1.0, 2.0, 4.0, 10.0]
burn_out_lengths = [1, 10, 50, 100]
quantiles = (.1, .9)

LOC_PARAM_IDX, SCALE_PARAM_IDX, NUM_PARAM_SITES = 0, 1, 2
MEAN_ERROR_IDX, STDERR_ERROR_IDX, QUANT_LOWER_IDX, QUANT_UPPER_IDX, RHAT_CROSS_IDX, RHAT_IN_IDX, NUM_METRICS = 0, 1, 2, 3, 4, 5, 6
params_errors = np.zeros((len(epsilons), NUM_PARAM_SITES, len(burn_out_lengths), NUM_METRICS))
from_chain_variance = np.zeros((len(epsilons), NUM_PARAM_SITES, len(burn_out_lengths), num_params))
cross_last_variance = np.zeros((len(epsilons), NUM_PARAM_SITES, num_params))

def squared_error(x, y): return (x-y)**2

trace_to_show = 23 # handpicked for good mixing (lowest Rhat) between the different DP runs

for eps_idx, epsilon in enumerate(epsilons):

        params_errors_for_eps = np.full((NUM_PARAM_SITES, len(burn_out_lengths), len(seeds), num_params), np.nan)
        cross_rhats_for_eps = np.full((NUM_PARAM_SITES, len(burn_out_lengths), num_params), np.nan)
        in_rhats_for_eps = np.full((NUM_PARAM_SITES, len(burn_out_lengths), len(seeds), num_params), np.nan)
        in_chain_vars_for_eps = np.full((NUM_PARAM_SITES, len(burn_out_lengths), len(seeds), num_params), np.nan)

        traces_per_seed = []
        last_params_for_eps = np.full((NUM_PARAM_SITES, len(seeds), num_params), np.nan)

        for seed_idx, seed in enumerate(seeds):
            run_traces = load_traces_for_run(
                args.stored_pickle_dir, args.prefix, epsilon, seed, args.clipping_threshold, args
            )
            run_traces = traces_to_dict(run_traces)
            traces_per_seed.append(run_traces)

            last_params_for_eps[LOC_PARAM_IDX, seed_idx] = run_traces['auto_loc'][-1]
            last_params_for_eps[SCALE_PARAM_IDX, seed_idx] = run_traces['auto_scale'][-1]

            for burn_out_idx, burn_out in enumerate(burn_out_lengths):
                run_avg_params, unconst_params_var, unconst_params_Rhat = average_params(
                    run_traces, burn_out
                )

                # avg_params[eps_idx, LOC_PARAM_IDX, burn_out_idx] = run_avg_params['auto_loc']
                # avg_params[eps_idx, SCALE_PARAM_IDX, burn_out_idx] = run_avg_params['auto_scale']
                avg_errors = jax.tree_multimap(
                    squared_error, run_avg_params, baseline_params
                )
                params_errors_for_eps[LOC_PARAM_IDX, burn_out_idx, seed_idx] = avg_errors['auto_loc']
                params_errors_for_eps[SCALE_PARAM_IDX, burn_out_idx, seed_idx] = avg_errors['auto_scale']

                if burn_out > 1:
                    split_Rhat_for_run = split_Rhat(run_traces, only_last=burn_out, num_splits=2, shuffle=False)
                    in_rhats_for_eps[LOC_PARAM_IDX, burn_out_idx, seed_idx] = split_Rhat_for_run['auto_loc']
                    in_rhats_for_eps[SCALE_PARAM_IDX, burn_out_idx, seed_idx] = split_Rhat_for_run['auto_scale']

                    run_trace_vars = jax.tree_map(lambda site_trace: np.var(site_trace[-burn_out:], ddof=1, axis=0), run_traces)
                    in_chain_vars_for_eps[LOC_PARAM_IDX, burn_out_idx, seed_idx] = run_trace_vars['auto_loc']
                    in_chain_vars_for_eps[SCALE_PARAM_IDX, burn_out_idx, seed_idx] = run_trace_vars['auto_scale']

        for burn_out_idx, burn_out in enumerate(burn_out_lengths):
            if burn_out > 1:
                rhats = multirun_Rhat(traces_per_seed, only_last=burn_out, runs_are_first=True)
                cross_rhats_for_eps[LOC_PARAM_IDX, burn_out_idx] = rhats['auto_loc']
                cross_rhats_for_eps[SCALE_PARAM_IDX, burn_out_idx] = rhats['auto_scale']

        from_chain_variance[eps_idx] = np.mean(in_chain_vars_for_eps, axis=-2)
        cross_last_variance[eps_idx] = np.var(last_params_for_eps, ddof=1, axis=-2)

        # # for single seed, single param trace
        # params_errors[eps_idx, :, :, MEAN_ERROR_IDX] = params_errors_for_eps[:, :, 0, trace_to_show]

        # averaged over seeds, single param trace
        params_errors[eps_idx, :, :, MEAN_ERROR_IDX] = np.mean(params_errors_for_eps[:, :, :, trace_to_show], axis=-1)
        params_errors[eps_idx, :, :, STDERR_ERROR_IDX] = np.std(params_errors_for_eps[:, :, :, trace_to_show], ddof=1, axis=-1) / np.sqrt(len(seeds))
        params_errors[eps_idx, :, :, [QUANT_LOWER_IDX, QUANT_UPPER_IDX]] = np.quantile(params_errors_for_eps[:, :, :, trace_to_show], quantiles, axis=-1)
        params_errors[eps_idx, :, :, RHAT_CROSS_IDX] = cross_rhats_for_eps[:, :, trace_to_show]
        params_errors[eps_idx, :, :, RHAT_IN_IDX] = in_rhats_for_eps[:, :, 0, trace_to_show] #np.mean(in_rhats_for_eps[:, :, :, trace_to_show], axis=-1)


import matplotlib.pyplot as plt
plt.rcParams.update(neurips_fig_style)
plt.rcParams.update({"font.size":12})
err_type_suffix = f'_{args.error_type}'
if args.error_type == 'stderr':
    def compute_yerr(plot_data):
            return plot_data[STDERR_ERROR_IDX]
else:
    def compute_yerr(plot_data):
            return (
                plot_data[MEAN_ERROR_IDX] - plot_data[QUANT_LOWER_IDX],
                plot_data[QUANT_UPPER_IDX] - plot_data[MEAN_ERROR_IDX]
            )


### PLOTTING MSE ERRORS
fig, axis = plt.subplots(nrows=2, sharex='col')

for burn_out_idx, burn_out in enumerate(burn_out_lengths):

    # # for single seed, single param
    # axis[0].plot(epsilons, params_errors[:, LOC_PARAM_IDX, burn_out_idx, MEAN_ERROR_IDX], alpha=.7, label=f'averaging last {burn_out}')
    # axis[1].plot(epsilons, params_errors[:, SCALE_PARAM_IDX, burn_out_idx, MEAN_ERROR_IDX], alpha=.7, label=f'averaging last {burn_out}')

    # averaging seed, single param
    axis[0].errorbar(x=epsilons, y=params_errors[:, LOC_PARAM_IDX, burn_out_idx, MEAN_ERROR_IDX], yerr=compute_yerr(params_errors[:, LOC_PARAM_IDX, burn_out_idx]), label=f'averaging last {burn_out}')
    axis[1].errorbar(x=epsilons, y=params_errors[:, SCALE_PARAM_IDX, burn_out_idx, MEAN_ERROR_IDX], yerr=compute_yerr(params_errors[:, SCALE_PARAM_IDX, burn_out_idx]), label=f'averaging last {burn_out}')


axis[0].set_xticks(epsilons)
axis[0].set_xticklabels(epsilons)
axis[0].set_xlabel("$\epsilon$")
axis[0].set_ylabel("MSE")
axis[0].set_title("Variational posterior means")

axis[1].set_xticks(epsilons)
axis[1].set_xticklabels(epsilons)
axis[1].set_xlabel("$\epsilon$")
axis[1].set_ylabel("MSE")
axis[1].set_title("Variational posterior scales")


fig.suptitle(f"Averaging traces for different final chain lengths\nMSE of obtained parameter estimate to nonprivate VI baseline result.\n {args.prefix} method , C={args.clipping_threshold}")
fig.legend(*(axis[0].get_legend_handles_labels()))

plot_name = filenamer(f"trace_averaging_MSE_{args.prefix}", args, seed="all", eps="alleps") + err_type_suffix
fig.savefig(f"{os.path.join(args.output_dir, plot_name)}.pdf")
plt.show()

############ PLOTTING RHAT ##############

fig, axis = plt.subplots(ncols=2, nrows=2, sharex=True, sharey='row')

for burn_out_idx, burn_out in enumerate(burn_out_lengths):
    # averaging seed, single param
    axis[0, 0].plot(epsilons, params_errors[:, LOC_PARAM_IDX, burn_out_idx, RHAT_CROSS_IDX], label=f'averaging last {burn_out}')
    axis[0, 1].plot(epsilons, params_errors[:, SCALE_PARAM_IDX, burn_out_idx, RHAT_CROSS_IDX], label=f'averaging last {burn_out}')

    axis[1, 0].plot(epsilons, params_errors[:, LOC_PARAM_IDX, burn_out_idx, RHAT_IN_IDX], label=f'averaging last {burn_out}')
    axis[1, 1].plot(epsilons, params_errors[:, SCALE_PARAM_IDX, burn_out_idx, RHAT_IN_IDX], label=f'averaging last {burn_out}')


axis[0, 0].set_xticks(epsilons)
axis[0, 0].set_xticklabels(epsilons)
axis[0, 0].set_xlabel("$\epsilon$")
axis[0, 0].set_ylabel("$\hat{R}$ over DP runs")
axis[0, 0].set_title("Variational posterior means")

axis[0, 1].set_xticks(epsilons)
axis[0, 1].set_xticklabels(epsilons)
axis[0, 1].set_xlabel("$\epsilon$")
axis[0, 1].set_title("Variational posterior scales")


axis[1, 0].set_xticks(epsilons)
axis[1, 0].set_xticklabels(epsilons)
axis[1, 0].set_xlabel("$\epsilon$")
axis[1, 0].set_ylabel("split-$\hat{R}$ in single chain (single DP run)")
# axis[1, 0].set_title("Variational posterior means")

axis[1, 1].set_xticks(epsilons)
axis[1, 1].set_xticklabels(epsilons)
axis[1, 1].set_xlabel("$\epsilon$")
# axis[1, 1].set_ylabel("$\hat{R}$")
# axis[1, 1].set_title("Variational posterior scales")


fig.suptitle("Averaging traces for different final chain lengths\n$\hat{R}$ between DP runs and split-$\hat{R}$ in single chain" + f"\n {args.prefix} method , C={args.clipping_threshold}")
fig.legend(*(axis[0,0].get_legend_handles_labels()))

plot_name = filenamer(f"trace_averaging_Rhat_{args.prefix}", args, seed="all", eps="alleps")
fig.savefig(f"{os.path.join(args.output_dir, plot_name)}.pdf")
plt.show()

# PLOTTING VARIANCE ESTIMATES FROM CHAINS
fig, axis = plt.subplots(nrows=2, sharex='col')

burn_out_idx = -1
for param_idx in [LOC_PARAM_IDX, SCALE_PARAM_IDX]:
# axis[0].plot(epsilons, from_chain_variance[:, LOC_PARAM_IDX, burn_out_idx, ])
    axis[param_idx].plot(epsilons, cross_last_variance[:, param_idx, trace_to_show], label="Variance from last epoch over DP runs")
    for burn_out_idx, burn_out in enumerate(burn_out_lengths):
        axis[param_idx].plot(epsilons, from_chain_variance[:, param_idx, burn_out_idx, trace_to_show], label=f"mean variance in length {burn_out}")

axis[0].set_xticks(epsilons)
axis[0].set_xticklabels(epsilons)
axis[0].set_xlabel("$\epsilon$")
axis[0].set_ylabel("Sample variance")
axis[0].set_title("Variational means")

axis[1].set_xticks(epsilons)
axis[1].set_xticklabels(epsilons)
axis[1].set_xlabel("$\epsilon$")
axis[1].set_ylabel("Sample variance")
axis[1].set_title("Variational scales")

fig.suptitle("Variance estimates from different final chain lengths (averaged) compared to variance of last epoch over DP runs")
fig.legend(*(axis[0].get_legend_handles_labels()))

plot_name = filenamer(f"trace_averaging_vars_{args.prefix}", args, seed="all", eps="alleps")
fig.savefig(f"{os.path.join(args.output_dir, plot_name)}.pdf")
plt.show()
