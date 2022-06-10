import argparse, os, sys, pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils import filenamer, fit_model1, make_names_pretty, neurips_fig_style

from collections import defaultdict, namedtuple

from pvalue_utils import pool_analysis

parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
parser.add_argument('data_path', type=str, help='Path to input data.')
parser.add_argument("stored_pickle_dir", type=str, help="Dir from which to read traces.")
parser.add_argument("nondp_pickle_dir", type=str, help="Dir from which to read nondp baseline traces")
parser.add_argument("--output_dir", type=str, default="./", help="Dir to store the results")
# parser.add_argument("--prefix", type=str, help="Type of a DPSVI")
def parse_clipping_threshold_arg(value): return None if value == "None" else float(value)
parser.add_argument("--clipping_threshold", default=None, type=parse_clipping_threshold_arg, help="Clipping threshold for DP-SGD.")
# parser.add_argument("--seed", default=None, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
parser.add_argument("--epsilon", default=1.0, type=float)
parser.add_argument("--k", default=16, type=int, help="Mixture components in fit (for automatic modelling only).")
parser.add_argument("--num_epochs", "-e", default=1000, type=int, help="Number of training epochs.")
parser.add_argument("--init_scale", type=float, default=0.1)
parser.add_argument("--plot_traces", action='store_true', default=False)
parser.add_argument("--plot_robustness", action='store_true', default=False)
parser.add_argument("--plot_robustness_tradeoff", action='store_true', default=False)
parser.add_argument("--plot_error_over_epochs", action='store_true', default=False)
parser.add_argument("--error_type", choices=['stderr', 'quantiles'], default='stderr')

args, unknown_args = parser.parse_known_args()

# og_prefix = args.prefix
# if og_prefix is not None:
#     if "init" in og_prefix:
#         init_scale = og_prefix.split("init")[1]

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
from metrics import init_proportional_abs_error

def load_traces_for_run(path, prefix, epsilon, seed, clipping_threshold, args):
    wholepop_output_name = filenamer(prefix, args, epsilon=epsilon, seed=seed, clipping_threshold=clipping_threshold)
    wholepop_output_path = f"{os.path.join(path, wholepop_output_name)}_traces.p"

    trace = pd.read_pickle(wholepop_output_path)
    loc_trace = guide._unpack_latent(trace.loc_trace)
    scale_trace = guide._unpack_latent(trace.scale_trace)

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


PREFIXES = ['vanilla', 'aligned', 'ng_adam-3', 'aligned_ng_adam-3']

### load DP results
stored_pickle_dir = args.stored_pickle_dir
if init_scale != 0.1:
    stored_pickle_dir = os.path.join(stored_pickle_dir, f"init{init_scale}/")

epsilons = [1.0, 2.0, 4.0, 10.0]
seeds = list(range(123, 133))

trace_errors_loc = dict()
trace_errors_scale = dict()

Cs = {'vanilla': 2.0, 'aligned':2.0, 'ng_adam-3':0.1, 'aligned_ng_adam-3':0.1}
# vanilla
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
fig, axis = plt.subplots(nrows=2)
plot_name = filenamer(f"trace_error_aligned_vs_vanilla_init{init_scale}", args, seed="all") + err_type_suffix
thinning = 1
T = args.num_epochs

COLORS = ["blue", "red"]
for prefix in PREFIXES:
    ## locs
    loc_plot_data = np.array(list(trace_errors_loc[prefix][args.epsilon].values()))[:, ::thinning]
    ts = np.arange(0, T)[::thinning]
    for dim, color in enumerate(COLORS):
        axis[0].errorbar(
                ts,
                loc_plot_data[:,:,dim].mean(0),
                compute_yerr(loc_plot_data[:,:,dim], axis=0),
                ls="--", color=color, alpha=0.2, label=f"{prefix}, dim={dim}"
        )

    ## scales
    scale_plot_data = np.array(list(trace_errors_scale[prefix][args.epsilon].values()))[:, ::thinning]
    for dim, color in enumerate(COLORS):
        axis[1].errorbar(
                ts,
                scale_plot_data[:,:,dim].mean(0),
                compute_yerr(scale_plot_data[:,:,dim], axis=0),
                ls="--", color=color, alpha=0.2
        )

#### average over all the parameters
fig, axis = plt.subplots(figsize=(6, 4), nrows=2, sharex=True)
plot_name = filenamer(f"avg_trace_error_aligned_vs_vanilla_init{init_scale}", args, seed="all") + err_type_suffix

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
Cs = {'vanilla': 2.0, 'aligned':2.0, 'ng_adam-3':0.1, 'aligned_ng_adam-3':0.1}
new_labels = []
for label in labels:
    if label == "ng_adam-3":
        new_labels.append("Natural grad.")
    elif label == "aligned_ng_adam-3":
        new_labels.append("Aligned natural grad")
    else:
        new_labels.append(label)

#axis[1].legend(handles, new_labels)
fig.legend(handles, new_labels, fontsize=7)


fig.suptitle(f'Evolution of error in learned parameters\n$\\varepsilon=${args.epsilon}', fontsize=11.5)

plt.tight_layout()

axis[0].set_yscale('log')
axis[1].set_yscale('log')
plot_name = f"avg_trace_error_{args.epsilon}_k16_seedall_epochs{args.num_epochs}_ng_comparison_init{args.init_scale}_stderr_logscale.pdf"
plt.savefig(plot_name, format="pdf", bbox_inches='tight')
plt.close()
