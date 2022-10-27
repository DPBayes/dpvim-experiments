"""
This script computes the log-likelihood on simulated test data for the inference
results produced by `linear_regression_infer.py` and creates the plot for
Figure 5.
"""

import jax, os

import pandas as pd
import numpy as np
import jax.numpy as jnp

from numpyro.distributions import Normal

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("result_dir")
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--clipping_threshold", type=float, default=2.0)
parser.add_argument("--corr_method", choices=['LKJ', 'manual'], default="LKJ")
args = parser.parse_args()

num_repeats = 50

from common import generate_corr_chol_manual, model, generate_data, generate_corr_chol_from_LKJ, generate_corr_chol_manual

from numpyro.infer.autoguide import AutoMultivariateNormal
from smart_auto_guide import SmartAutoGuide

ll_df = pd.DataFrame()
ll_diff_df = pd.DataFrame()
offdiags_df = pd.DataFrame()

from itertools import product
if args.corr_method == "LKJ":
    corr_strengths = [0.1, 1000.]
else:
    corr_strengths = [0.2, 0.8]
num_dimss = [100, 200]
epsilons = [1., 2.]
variants = ['aligned_fullrank', 'vanilla_fullrank', 'nondp_fullrank']
seeds = range(123, 123 + num_repeats)
idx = pd.MultiIndex.from_product(
    [variants, epsilons, num_dimss, corr_strengths, seeds],
    names=['variant', 'eps', 'ndims', 'corr', 'seed']
)

parameters = pd.DataFrame({'params': pd.Series(dtype=object)}, index=idx)

for corr_strength, num_dims in product(corr_strengths, num_dimss):
    print(f"{corr_strength=}, {num_dims=}")
    clip = args.clipping_threshold


    ### Generate data
    n_train = 10000
    n_test = 10000
    d_without_intercept = num_dims

    np_rng = np.random.RandomState(seed=123)
    if args.corr_method == "LKJ":
        L = generate_corr_chol_from_LKJ(np_rng, d_without_intercept, corr_strength)
    else:
        L = generate_corr_chol_manual(np_rng, d_without_intercept, corr_strength, .8)
    X_train, y_train, true_weight = generate_data(np_rng, L, n_train)
    X_test, y_test, _ = generate_data(np_rng, L, n_test, true_weight=true_weight)
    print("max abs offdiag elem from covariance matrix: ", np.abs(np.tril(L @ L.T, -1)).max())

    offdiag_covs = np.abs(np.tril(L @ L.T, -1)).ravel()
    offdiags_df = pd.concat((
            offdiags_df,
            pd.DataFrame({'c': offdiag_covs, 'corr_strength': np.zeros_like(offdiag_covs) + corr_strength})
        ),
        ignore_index=True
    )

    guide = SmartAutoGuide.wrap_for_sampling(AutoMultivariateNormal, ("X", "y"))(model)

    guide.initialize(X_train, y_train)

    def predictive_loglikelihood(params):
        unpacked_means = guide.base_guide._unpack_and_constrain(params["auto_loc"], params)
        posterior_cov_chol = params["auto_scale_tril"]
        indices_for_weight = guide._unpack_latent(np.arange(len(params["auto_loc"])))["weight"].astype(int)
        weight_posterior_cov_chol_indices = np.meshgrid(indices_for_weight, indices_for_weight, indexing="ij")
        weight_posterior_cov_chol = posterior_cov_chol[tuple(weight_posterior_cov_chol_indices)]
        weight_posterior_cov = weight_posterior_cov_chol @ weight_posterior_cov_chol.T
        pred_mean = X_test @ unpacked_means["weight"]
        pred_var = (X_test * (X_test @ weight_posterior_cov)).sum(1) + unpacked_means["noise_std"]**2
        return Normal(pred_mean, np.sqrt(pred_var)).log_prob(y_test), pred_mean, pred_var



    ### load the trained models
    for epsilon in epsilons:

        def load_results(variant, result_df):
            fname = os.path.join(
                        args.result_dir,
                        f"linreg_params_corr{corr_strength}_ndim{num_dims}_{variant}_eps{epsilon}_clip{clip}_seed{seed}.p"
                    )
            if os.path.exists(fname):
                result_df.loc[variant, epsilon, num_dims, corr_strength, seed]["params"] = pd.read_pickle(fname)
                return result_df

        for seed in range(123, 123 + num_repeats):
            for variant in variants:
                load_results(variant, parameters)

    def predictive_avg_loglikelihood(params):
        return predictive_loglikelihood(params)[0].mean().item()

    mean_pred_ll = parameters.loc[:, :, num_dims, corr_strength, :].dropna().apply(lambda x: predictive_avg_loglikelihood(x['params']), axis='columns').reset_index()
    mean_pred_ll['ndims'] = num_dims
    mean_pred_ll['corr'] = corr_strength
    ll_df = pd.concat([ll_df, mean_pred_ll])

ll_df = ll_df.set_index(['variant', 'eps', 'ndims', 'corr', 'seed'])
ll_df.columns = ['ll']

import matplotlib.pyplot as plt
import seaborn as sns

def filter_df(df, columns_and_values, negate=False):
    idxs = np.ones_like(df.index, dtype=bool)
    if isinstance(columns_and_values, dict):
        columns_and_values_dict = columns_and_values
    else:
        columns_and_values_dict = dict()
        for column, values in columns_and_values:
            columns_and_values_dict[column] = values

    if not isinstance(negate, dict):
        negate_dict = dict()
        if isinstance(negate, (list, tuple)):
            for column in negate:
                negate_dict[column] = True
        elif isinstance(negate, bool):
            for column in columns_and_values_dict.keys():
                negate_dict[column] = negate
    else:
        negate_dict = negate

    for column, values in columns_and_values_dict.items():
        column_idxs = np.zeros_like(df.index, dtype=bool)
        if isinstance(values, str):
            values = (values,)
        for value in values:
            column_idxs = column_idxs | (df[column] == value)
        column_idxs ^= negate_dict.get(column, False)
        idxs &= column_idxs

    return df[idxs]

default_aspect = 12. / 9.
neurips_fig_style = {'figure.figsize':(5.5, 5.5 / default_aspect), 'font.size':10}
plt.rcParams.update(neurips_fig_style)

# FIGURE 5
# plot vanilla and aligned for concentrations and eps=1.
epsilons = [1.]
num_epsilons = len(epsilons)
num_corr_strengths = len(corr_strengths)

fig, axes = plt.subplots(num_epsilons, num_corr_strengths, sharex=True, sharey=True)
if num_epsilons == 1:
    axes = np.array([axes])

for i, eps in enumerate(epsilons):
    for j, corr_strength in enumerate(corr_strengths):
        ax = axes[i, j]

        plot_df = ll_df.loc[['vanilla_fullrank', 'aligned_fullrank'], eps, :, corr_strength].reset_index()
        sns.boxplot(x='ndims', y='ll', hue='variant', hue_order=["vanilla_fullrank", "aligned_fullrank"], data=plot_df, ax=ax)

        nondp_mean = ll_df.loc['nondp_fullrank', eps, num_dims].mean().item()
        ax.axhline(nondp_mean, ls='--', c='k', alpha=.8, label='nondp')


        ax.set_xlabel(None)
        if num_epsilons > 1 and j == 0:
            ax.set_ylabel(f"{eps=}")
        else:
            ax.set_ylabel(None)

        if i == 0:
            if args.corr_method == "LKJ":
                ax.set_title(f'concentration {corr_strength}')
            else:
                ax.set_title(f'density {corr_strength}')
        ax.get_legend().set_visible(False)

fig.suptitle("Predictive log-likelihood for full-rank approximation")
fig.supylabel("average predictive log-likelihood")
fig.supxlabel("number of dimensions")
fig.legend(*axes[0,0].get_legend_handles_labels(), loc='lower left')
fig.tight_layout()
fig.savefig(f"linreg_logp_box_{args.corr_method}corr.pdf", format="pdf")
plt.close()
