"""
This script is used to plot Figure 3 (as well as various figures in the supplement).
"""


import argparse, os, sys, pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from utils import neurips_fig_style
plt.rcParams.update(neurips_fig_style)
import seaborn as sns

from utils import filenamer, fit_model1, make_names_pretty, pool_analysis

from collections import defaultdict

parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
parser.add_argument("stored_pickle_dir", type=str, help="Dir from which to read learned downstream results.")
parser.add_argument("--data_path", type=str)
parser.add_argument("--output_dir", type=str, default="./", help="Dir to store the results")
parser.add_argument("--prefix", type=str, default="aligned", help="Type of a DPSVI")
parser.add_argument("--clipping_threshold", default=2., type=float, help="Clipping threshold for DP-SGD.")
parser.add_argument("--k", default=16, type=int, help="Mixture components in fit (for automatic modelling only).")
parser.add_argument("--num_epochs", "-e", default=2000, type=int, help="Number of training epochs.")
parser.add_argument("--force_recompute", action='store_true', default=False)
# parser.add_argument("--plot_confint_scatter", action='store_true', default=False)
parser.add_argument("--plot_l2s", action='store_true', default=False)
parser.add_argument("--plot_l2s_converged", action='store_true', default=False)

args, unknown_args = parser.parse_known_args()

### fit the original model
## read the whole UKB data
try:
    orig_summary_df= pd.read_pickle(os.path.join(args.stored_pickle_dir, "downstream_results_orig_data.p"))
except FileNotFoundError:
    df_whole = pd.read_csv(args.data_path)
    print("Loaded data set has {} rows (entries) and {} columns (features).".format(*df_whole.shape))
    orig_statsmodels_fit, _ = fit_model1(df_whole)
    orig_summary_df = pd.read_html(orig_statsmodels_fit.summary().tables[1].as_html(), index_col=0, header=0)[0].rename(make_names_pretty)

    pd.to_pickle(orig_summary_df, os.path.join(args.stored_pickle_dir, "downstream_results_orig_data.p"))

ethnicity_keys = [name for name in orig_summary_df.index if "ethnicity" in name]
ethnicity_names = [name.split(":")[1].strip() for name in orig_summary_df.index if "ethnicity" in name]
n_eth = len(ethnicity_names)

### read synthetic data results
epsilons = [1.0, 2.0, 4.0, 10.0]
seeds = list(range(123, 133))

args.seed = "all"
args.epsilon = "alleps"
og_prefix = args.prefix

burn_out_lengths = [500, 100, 20]


# else:
if True:
    args.seed = "all"
    args.epsilon = "alleps"
    args.prefix = og_prefix

    # only last iterate
    array_of_coef_estimates_only_last = defaultdict(dict)
    array_of_coef_variances_only_last = defaultdict(dict)
    for epsilon in epsilons:
        args.epsilon = epsilon
        for seed in seeds:
            args.seed = seed
            output_name = "downstream_results_" + filenamer(args.prefix, "", args)
            output_path = f"{os.path.join(args.stored_pickle_dir, output_name)}.p"

            results_df_only_last = pd.read_pickle(output_path)

            # aggregate M synthetic datas using Rubins rules
            array_of_coef_estimates_only_last[epsilon][seed] = pd.concat([res["coef"] for res in results_df_only_last], axis=1)
            array_of_coef_variances_only_last[epsilon][seed] = pd.concat([res["std err"]**2 for res in results_df_only_last], axis=1)

    # averaged
    array_of_coef_estimates_averaged = defaultdict(lambda: defaultdict(dict))
    array_of_coef_variances_averaged = defaultdict(lambda: defaultdict(dict))
    burn_outs_to_load = []
    if args.plot_l2s:
        burn_outs_to_load += burn_out_lengths
    if args.plot_l2s_converged:
        burn_outs_to_load.append("converged")
    for burn_out in burn_outs_to_load:
        args.prefix = f"avg{burn_out}_{og_prefix}"
        for epsilon in epsilons:
            args.epsilon = epsilon
            for seed in seeds:
                args.seed = seed
                output_name = "downstream_results_" + filenamer(args.prefix, "", args)
                output_path = f"{os.path.join(args.stored_pickle_dir, output_name)}.p"

                results_df_averaged = pd.read_pickle(output_path)

                # aggregate M synthetic datas using Rubins rules
                array_of_coef_estimates_averaged[epsilon][burn_out][seed] = pd.concat([res["coef"] for res in results_df_averaged], axis=1)
                array_of_coef_variances_averaged[epsilon][burn_out][seed] = pd.concat([res["std err"]**2 for res in results_df_averaged], axis=1)

    ### apply the rubins rules to the coefficients and store
    only_last_rubins_Qm = defaultdict(dict)
    only_last_rubins_Tm = defaultdict(dict)
    for epsilon in epsilons:
        for seed in seeds:
            Q_m, T_m = pool_analysis(array_of_coef_estimates_only_last[epsilon][seed], array_of_coef_variances_only_last[epsilon][seed], return_pvalues=False)
            only_last_rubins_Qm[epsilon][seed] = Q_m
            only_last_rubins_Tm[epsilon][seed] = T_m

    averaged_rubins_Qm = dict()
    averaged_rubins_Tm = dict()
    for epsilon in epsilons:
        averaged_rubins_Qm[epsilon] = dict()
        averaged_rubins_Tm[epsilon] = dict()
        for burn_out in burn_outs_to_load:
            averaged_rubins_Qm[epsilon][burn_out] = dict()
            averaged_rubins_Tm[epsilon][burn_out] = dict()
            for seed in seeds:
                Q_m, T_m = pool_analysis(array_of_coef_estimates_averaged[epsilon][burn_out][seed], array_of_coef_variances_averaged[epsilon][burn_out][seed], return_pvalues=False)
                averaged_rubins_Qm[epsilon][burn_out][seed] = Q_m
                averaged_rubins_Tm[epsilon][burn_out][seed] = T_m

args.prefix = og_prefix

# PLOT the l2 errors in downstream coefficients over epsilon per burn out length
if args.plot_l2s:
    orig_Qms = np.array(orig_summary_df['coef'])
    orig_Tms = np.array(orig_summary_df['std err'])**2

    fig, ax = plt.subplots()
    Qms_l2_errors = []
    for epsilon in epsilons:
        Qms_as_arr = np.array(list(only_last_rubins_Qm[epsilon].values()))
        Qms_l2_errors.append(np.linalg.norm(Qms_as_arr - orig_Qms, axis=-1))
    Qms_l2_errors = np.array(Qms_l2_errors)
    ax.errorbar(
        epsilons, Qms_l2_errors.mean(axis=-1),
        Qms_l2_errors.std(axis=-1, ddof=1) / np.sqrt(Qms_l2_errors.shape[0]),
        capsize=2,
        alpha=.7, label=f'last iterate'
    )

    for burn_out in burn_out_lengths:
        Qms_l2_errors = []
        for epsilon in epsilons:
            Qms_as_arr = np.array(list(averaged_rubins_Qm[epsilon][burn_out].values()))
            # Tms_as_arr = np.array(list(averaged_rubins_Tm[epsilon][burn_out].values()))

            Qms_l2_errors.append(np.linalg.norm(Qms_as_arr - orig_Qms, axis=-1))

        Qms_l2_errors = np.array(Qms_l2_errors)
        ax.errorbar(
            epsilons, Qms_l2_errors.mean(axis=-1),
            Qms_l2_errors.std(axis=-1, ddof=1) / np.sqrt(Qms_l2_errors.shape[0]),
            capsize=2,
            alpha=.7, label=f'averaged {burn_out}'
        )

    ax.set_xlabel('$\\varepsilon$')
    ax.set_ylabel("$\ell_2$ error of downstream coefficients on synthetic data")
    ax.legend()
    fig.suptitle(f"Averaging effect in downstream estimation")
    args.seed = "all"
    args.epsilon = "all"
    plot_name = filenamer(f"avg_error_downstream", f"{args.prefix}", args) + ".pdf"
    fig.savefig(os.path.join(args.output_dir, plot_name))
    plt.close()

    boxplot_df = pd.DataFrame()
    for epsilon in epsilons:
        Qms_as_arr = np.array(list(only_last_rubins_Qm[epsilon].values()))
        Qms_l2_err = np.linalg.norm(Qms_as_arr - orig_Qms, axis=-1)
        boxplot_df = pd.concat((boxplot_df,
            pd.DataFrame({
                'epsilon': epsilon,
                'seed': only_last_rubins_Qm[epsilon].keys(),
                'l2': Qms_l2_err,
                'Method': 'Last',
            })
        ))

        for burn_out in sorted(burn_out_lengths):
            Qms_as_arr = np.array(list(averaged_rubins_Qm[epsilon][burn_out].values()))
            Qms_l2_err = np.linalg.norm(Qms_as_arr - orig_Qms, axis=-1)
            boxplot_df = pd.concat((boxplot_df,
                pd.DataFrame({
                    'epsilon': epsilon,
                    'seed': averaged_rubins_Qm[epsilon][burn_out].keys(),
                    'l2': Qms_l2_err,
                    'Method': f'Avg. {burn_out}',
                })
            ))


    ax = sns.boxplot(x='epsilon', y='l2', hue='Method', data=boxplot_df)
    ax.set_xlabel('$\\varepsilon$')
    ax.set_ylabel("$\ell_2$ error of downstream coefficients on synthetic data")
    ax.set_title('Effect of iterate averaging on downstream coefficients')

    plot_name = filenamer(f"avg_error_downstream_boxplot", f"{args.prefix}", args) + ".pdf"
    plt.savefig(os.path.join(args.output_dir, plot_name))
    plt.close()



# PLOT the l2 errors in downstream coefficients over epsilon where averages are over chains considered converged
if args.plot_l2s_converged:
    orig_Qms = np.array(orig_summary_df['coef'])
    orig_Tms = np.array(orig_summary_df['std err'])**2

    burn_out = "converged"
    boxplot_df = pd.DataFrame()
    for epsilon in epsilons:
        Qms_as_arr = np.array(list(only_last_rubins_Qm[epsilon].values()))
        Qms_l2_err = np.linalg.norm(Qms_as_arr - orig_Qms, axis=-1)
        boxplot_df = pd.concat((boxplot_df,
            pd.DataFrame({
                'epsilon': epsilon,
                'seed': only_last_rubins_Qm[epsilon].keys(),
                'l2': Qms_l2_err,
                'Method': 'Last',
            })
        ))

        Qms_as_arr = np.array(list(averaged_rubins_Qm[epsilon][burn_out].values()))
        Qms_l2_err = np.linalg.norm(Qms_as_arr - orig_Qms, axis=-1)
        boxplot_df = pd.concat((boxplot_df,
            pd.DataFrame({
                'epsilon': epsilon,
                'seed': averaged_rubins_Qm[epsilon][burn_out].keys(),
                'l2': Qms_l2_err,
                'Method': f'Averaged',
            })
        ))


    ax = sns.boxplot(x='epsilon', y='l2', hue='Method', data=boxplot_df)
    ax.set_xlabel('$\\varepsilon$')
    ax.set_ylabel("$\ell_2$ error of downstream coefficients on synthetic data")
    ax.set_title('Effect of iterate averaging on downstream coefficients')

    plot_name = filenamer(f"avg_converged_error_downstream_boxplot", f"{args.prefix}", args) + ".pdf"
    plt.savefig(os.path.join(args.output_dir, plot_name))
    plt.close()
