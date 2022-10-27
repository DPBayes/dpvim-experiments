"""
Contains various helper/utility functions.
"""

import re
import numpy as np
import pandas as pd
import argparse
import scipy.stats as stats


def filenamer(prefix, postfix, args=None, **kwargs):
    def filenamer_explicit(prefix, postfix, epsilon=None, clipping_threshold=None, k=None, seed=None, num_epochs=None, **unused_kwargs):
        output_name=f"{prefix}_{epsilon}_C{clipping_threshold}_k{k}_seed{seed}_epochs{num_epochs}_{postfix}"
        return output_name

    if isinstance(args, argparse.Namespace):
        new_kwargs = args.__dict__.copy()

        new_kwargs.update(kwargs)
        if 'prefix' in new_kwargs:
            del new_kwargs['prefix']
        kwargs = new_kwargs

    return filenamer_explicit(prefix, postfix, **kwargs)


# STATSMODELS for downstream inference
import statsmodels.api as sm
import statsmodels.formula.api as smf
statsmodels_formula = "covid_test_result ~ "\
    "C(age_group, Treatment(reference='[40, 45)')) + " \
    "C(sex, Treatment(reference='Female')) + " \
    "C(ethnicity, Treatment(reference='White British')) + "\
    "C(deprivation, Treatment(reference='1st')) + " \
    "C(education, Treatment(reference='College or University degree')) + " \
    "C(assessment_center, Treatment(reference='Newcastle'))"

def fit_model1(data):
    data = data.copy()

    data['covid_test_result'] = data['covid_test_result'].astype(int)
    model1 = smf.glm(formula=statsmodels_formula, family=sm.families.Poisson(), data=data)
    res = model1.fit(cov_type='HC1')
    return res, model1

def format_statsmodels_summary_as_df(summary):
    data = np.array(summary.tables[1].data)
    index = data[1:, 0]
    column_names = data[0, 1:]
    bulk_data = data[1:, 1:].astype(float)
    return pd.DataFrame(bulk_data, index=index, columns=column_names)

def make_names_pretty(name):
    if "reference" in name:
        feature, group = re.search("C\((\w+).*T\.(.*)]", name).groups()
        return f"{feature}: {group}"
    elif "T." in name:
        feature, group = re.search("(\w+).*T\.(.*)]", name).groups()
        return f"{feature}: {group}"
    else:
        return name

from collections import namedtuple
traces = namedtuple('traces', ['loc_trace', 'scale_trace'])

def traces_to_dict(trace_tuple: traces):
    return {
        'auto_loc': trace_tuple.loc_trace,
        'auto_scale': trace_tuple.scale_trace
    }

import os
import pickle
from numpyro.infer.autoguide import AutoDiagonalNormal, biject_to
from numpyro.distributions.transforms import IdentityTransform

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def single_param_site_single_linear_fit(param_trace, only_last=None):
    """
    Args:
        param_trace: A numpy array of shape (num_iterations, ...).
    """
    if isinstance(only_last, int):
        param_trace = param_trace[-only_last:]
    if np.any(np.isnan(param_trace)):
        return np.full(param_trace.shape[-1], np.nan), np.full(param_trace.shape[-1], np.nan)
    x = np.expand_dims(np.linspace(0., 1., len(param_trace)), -1)
    fit = LinearRegression().fit(x, param_trace)
    y_pred = fit.predict(x)
    r2 = r2_score(param_trace, y_pred, multioutput='raw_values')
    rmse = np.sqrt(np.mean((param_trace - y_pred)**2, axis=0))
    return fit.coef_.flatten(), fit.intercept_.flatten(), r2, rmse

def single_param_site_linear_fit(param_trace, only_last=None, spacing=None):
    """
    Args:
        param_trace: A numpy array of shape (num_iterations, ...).
    """
    if isinstance(only_last, int):
        param_trace = param_trace[-only_last:]
    if spacing is None:
        spacing = len(param_trace) // 10
    coefs = []
    intercepts = []
    r2s = []
    rmses = []
    if np.any(np.isnan(param_trace)):
        coefs = np.nan * np.ones((len(np.arange(0, len(param_trace), spacing)), param_trace.shape[-1]))
        intercepts = np.nan * np.ones((len(np.arange(0, len(param_trace), spacing)), param_trace.shape[-1]))
        return coefs, intercepts
    for start_indx in np.arange(0, len(param_trace), spacing):
        fit = single_param_site_single_linear_fit(param_trace[start_indx:])
        coefs.append(fit[0])
        intercepts.append(fit[1])
        r2s.append(fit[2])
        rmses.append(fit[3])
        # x = np.expand_dims(np.linspace(0., 1., len(param_trace[start_indx:])), -1)
        # fit = LinearRegression().fit(x, param_trace[start_indx:])
        # coefs.append(fit.coef_.flatten())
        # intercepts.append(fit.intercept_.flatten())
    return np.array(coefs), np.array(intercepts), r2s, rmses
    #return fit.coef_.flatten(), fit.intercept_


from typing import Union, Dict
def average_params(trace_tuple_or_dict: Union[traces, Dict[str, np.ndarray]], burn_out, transforms={'auto_loc': IdentityTransform(), 'auto_scale': biject_to(AutoDiagonalNormal.scale_constraint)}):
    """

    Returns (params, param_unconst_var):
        - params: Parameters averaged over the last `burn_out` values in unconstrained optimisation space, then transformed into constrained parameter space.
        - param_unconst_var: Parameter variances in unconstrained optimisation space.
    """

    if isinstance(trace_tuple_or_dict, traces):
        trace_dict = traces_to_dict(trace_tuple_or_dict)
    elif isinstance(trace_tuple_or_dict, dict):
        trace_dict = trace_tuple_or_dict
    else: raise ValueError(f"trace_tuple_or_dict must be a traces tuple or a dict; was {type(trace_tuple_or_dict)}.")

    params = {k: transforms[k](np.mean(transforms[k].inv(v[-burn_out:]), axis=0)) for k, v in trace_dict.items()}

    if burn_out > 1:
        param_unconst_var = {k: np.var(v[-burn_out], axis=0) for k, v in trace_dict.items()}
    else:
        param_unconst_var = {k: np.nan * np.ones(np.shape(v)[1:]) for k, v in trace_dict.items()}


    return params, param_unconst_var

def average_converged_params(trace_tuple_or_dict: Union[traces, Dict[str, np.ndarray]], spacing=100, threshold=0.05, transforms={'auto_loc': IdentityTransform(), 'auto_scale': biject_to(AutoDiagonalNormal.scale_constraint)}):
    if isinstance(trace_tuple_or_dict, traces):
        trace_dict = traces_to_dict(trace_tuple_or_dict)
    elif isinstance(trace_tuple_or_dict, dict):
        trace_dict = trace_tuple_or_dict
    else: raise ValueError(f"trace_tuple_or_dict must be a traces tuple or a dict; was {type(trace_tuple_or_dict)}.")

    params = dict()
    for site, param_trace in trace_dict.items():
        param_trace = transforms[site].inv(param_trace)
        coefs, _, r2s, rmses = single_param_site_linear_fit(param_trace, spacing=spacing)
        assert coefs.shape[-1] == param_trace.shape[-1]
        first_converged_idxs = np.full(coefs.shape[-1], -1, dtype=np.int64)
        params_at_site = np.empty(coefs.shape[-1])
        for i in range(coefs.shape[-1]):
            converged_indices = np.where(coefs[:,i] < threshold)[0]
            if len(converged_indices) > 1:
                first_converged_idxs[i] = converged_indices[0] * spacing
                params_at_site[i] = np.mean(param_trace[first_converged_idxs[i]:, i])
        params[site] = transforms[site](params_at_site)

    return params

def load_params_converged(prefix, postfix, args, spacing=400, threshold=0.05, **kwargs):
    stored_model_name = filenamer(prefix, postfix, args, **kwargs)
    stored_model_path = f"{os.path.join(args.stored_model_dir, stored_model_name)}"

    traces_path = f"{stored_model_path}_traces.p"
    if os.path.exists(traces_path): # always read from traces file, if it exists
        with open(traces_path, "rb") as f:
            trace_tuple_or_dict = pickle.load(f) # type: traces
    else: # error if traces file does not exists and average over last epochs is requested
        raise ValueError(f"You asked to average over converged burn out, but no traces file is available at {traces_path}")

    return average_converged_params(trace_tuple_or_dict, spacing=spacing, threshold=threshold)

def load_params(prefix, postfix, args, burn_out=None, **kwargs):
    """
    Loads parameters and averages in unconstrained space over the last `burn_out` (or `args.avg_over` if `burn_out` is None)
    epochs and compute their variance.

    Convenience wrapper for `average_params` that takes care of loading the file as well.

    Will always prioritize loading the corresponding traces pickle file, but falls back to loading the twinify
    output if no traces pickle file is available and `burn_out` (or `args.avg_over` if `burn_out` is None) equals 1.

    Returns (params, param_unconst_var):
        - params: Parameters averaged in unconstrained optimisation space, then transformed into constrained parameter space.
        - param_unconst_var: Parameter variances in unconstrained optimisation space.
    """
    if burn_out is None:
        burn_out = args.avg_over
    stored_model_name = filenamer(prefix, postfix, args, **kwargs)
    stored_model_path = f"{os.path.join(args.stored_model_dir, stored_model_name)}"

    traces_path = f"{stored_model_path}_traces.p"
    if os.path.exists(traces_path): # always read from traces file, if it exists
        with open(traces_path, "rb") as f:
            trace_tuple_or_dict = pickle.load(f) # type: traces

    elif args.avg_over == 1: # read from twinify output, if traces file does not exist but we are interested only in last iterand anyways
        with open(f"{stored_model_path}.p", "rb") as f:
            twinify_result = pickle.load(f)
        trace_tuple_or_dict = {
            'auto_loc': np.expand_dims(twinify_result.model_params['auto_loc'], 0),
            'auto_scale': np.expand_dims(twinify_result.model_params['auto_scale'], 0)
        }

    else: # error if traces file does not exists and average over last epochs is requested
        raise ValueError(f"You asked to average over {args.avg_over} last epochs, but no traces file is available at {traces_path}")

    return average_params(trace_tuple_or_dict, burn_out)


def init_proportional_abs_error(dp_trace, nondp_trace, average=True):
    """ MPAE! """
    baseline = nondp_trace[-1]
    scale = np.abs(dp_trace[0]-baseline)
    errors = np.abs(dp_trace-baseline)
    scaled_errors = errors / scale
    if average:
        return scaled_errors.mean(1) # average over parameters
    else:
        return scaled_errors

def pool_analysis(parameters, covariances, return_pvalues=True):
    """
    Pools results of downstream analysis from multiple synthetic data sets using Rubin's rules.

    parameters: ndarray of size (d, M) of estimators computed from the inputed data, where M is the number
                    of imputations
    covariances: list of covariance estimators associated with the parameters (same shape)
    """
    index = parameters.index
    # The mean estimator
    param_mean = parameters.mean(axis=1)

    # The average of the within-imputation covariances
    m = parameters.shape[1]
    cov_within = np.mean(covariances, axis=1)

    # The between-imputation covariance
    cov_between = np.var(parameters, axis=1)

    ## compute the p-value from the multi-dimensional Wald test
    Q_m = param_mean
    U_m = cov_within
    B_m = cov_between

    # compute the variance estimator
    T_m_pos = (1 + 1/m) * B_m + U_m # always positive, conservative
    T_m = (1 + 1/m) * B_m - U_m # can be negative
    T_m = np.where(T_m > 0, T_m, T_m_pos) # pick the conservative estimate for negative variances
    assert(np.all(T_m > 0))

    # compute pooled p-values
    v = (m - 1.)*(1. - m * U_m / ((m+1)*B_m))**2 # degree of freedom
    #pvalues = stats.f(1, v).sf(Q_m**2/T_m) # p-values from f-distribution
    pvalues = stats.chi2(1, v).sf(Q_m**2/T_m) # p-values from chi-squared.  Double check which one statsmodels uses

    #return pd.Series(pvalues, index=index), Q_m, T_m
    if return_pvalues:
        return (Q_m, pd.Series(T_m, index=index)), pd.Series(pvalues, index=index)
    else: return Q_m, pd.Series(T_m, index=index)

### Figure styles
default_aspect = 12. / 9.
neurips_fig_style = {'figure.figsize':(5.5, 5.5 / default_aspect), 'font.size':10}
