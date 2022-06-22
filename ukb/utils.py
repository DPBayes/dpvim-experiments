import re
import numpy as np
import argparse

def filenamer(prefix, args=None, **kwargs):
    def filenamer_explicit(prefix, epsilon=None, clipping_threshold=None, k=None, seed=None, num_epochs=None, **unused_kwargs):
        output_name=f"{prefix}_{epsilon}_C{clipping_threshold}_k{k}_seed{seed}_epochs{num_epochs}"
        return output_name

    if isinstance(args, argparse.Namespace):
        new_kwargs = args.__dict__.copy()
        new_kwargs.update(kwargs)
        if 'prefix' in new_kwargs:
            del new_kwargs['prefix']
        kwargs = new_kwargs

    return filenamer_explicit(prefix, **kwargs)


# STATSMODELS for downstream inference
import statsmodels.api as sm
import statsmodels.formula.api as smf
wholepop_formula = "covid_test_result ~ "\
    "C(age_group, Treatment(reference='[40, 45)')) + " \
    "C(sex, Treatment(reference='Female')) + " \
    "C(ethnicity, Treatment(reference='White British')) + "\
    "C(deprivation, Treatment(reference='1st')) + " \
    "C(education, Treatment(reference='College or University degree')) +"\
    "C(assessment_center, Treatment(reference='Newcastle'))"

def fit_model1(data):
    data = data.copy()
    formula = wholepop_formula

    data['covid_test_result'] = data['covid_test_result'].astype(int)
    model1 = smf.glm(formula=formula, family=sm.families.Poisson(), data=data)
    res = model1.fit(cov_type='HC1')
    return res, model1

def make_names_pretty(name):
    if "reference" in name:
        feature, group = re.search("C\((\w+).*T\.(.*)]", name).groups()
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

def multirun_Rhat(param_traces, only_last=None, runs_are_first=False, transforms={'auto_loc': IdentityTransform(), 'auto_scale': biject_to(AutoDiagonalNormal.scale_constraint)}):
    """
    Args:
        param_traces: Traces as (j)np arrays for each parameter site for each run
        only_last (int): Optional. Only consider the last `only_last` iterations from each trace. If None, Rhat is computed
            over the entire given trace.
        runs_are_first: If True, `unconstrained_param_traces` is treated as an iterable over dicts of parameter sites, i.e.,
            its type is assuemd to be Iterable[Dict[Any, np.ndarray]]. If False, the first two dimensions are reversed, i.e.
            its type is assumed to be Dict[Any, Iterable[np.ndarray]].
    """
    def multirun_Rhat_single_param_site(traces):
        assert traces.ndim >= 2
        if only_last is not None:
            traces = traces[:, -only_last:]

        M = traces.shape[0]
        N = traces.shape[1]

        mean_within_run_variance = np.mean(np.var(traces, ddof=1, axis=1), axis=0)
        between_run_variance_div_N = np.var(np.mean(traces, axis=1), ddof=1, axis=0)
        marginal_posterior_variance = ((N - 1) / N) * mean_within_run_variance + between_run_variance_div_N

        Rhat = np.sqrt(marginal_posterior_variance / mean_within_run_variance)
        return Rhat

    if runs_are_first:
        reorganised_traces = {
            site: np.array([runs[site] for runs in param_traces])
            for site in param_traces[0]
        }
    else:
        reorganised_traces = {
            site: np.array(list(values.values())) for site, values in param_traces.items()
        }
    return {
        site: multirun_Rhat_single_param_site(transforms[site].inv(values)) for site, values in reorganised_traces.items()
    }

def split_Rhat(param_traces, num_splits=2, only_last=None, shuffle=True, transforms={'auto_loc': IdentityTransform(), 'auto_scale': biject_to(AutoDiagonalNormal.scale_constraint)}):
    def split_Rhat_single_param_site(trace):
        assert trace.ndim >= 1
        if only_last is not None:
            trace = trace[-only_last:]
        if shuffle:
            trace = np.random.permutation(trace)
        trace_splits = np.split(trace, num_splits, axis=0)
        N = len(trace_splits[0])

        mean_within_split_variance = np.mean(np.var(trace_splits, ddof=1, axis=1), axis=0)
        between_split_variance_div_N = np.var(np.mean(trace_splits, axis=1), ddof=1, axis=0)
        marginal_posterior_variance = ((N - 1) / N) * mean_within_split_variance + between_split_variance_div_N

        Rhat = np.sqrt(marginal_posterior_variance / mean_within_split_variance)
        return Rhat

    return {
        site: split_Rhat_single_param_site(transforms[site].inv(values)) for site, values in param_traces.items()
    }

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
def average_params(trace_tuple_or_dict: Union[traces, Dict[str, np.ndarray]], burn_out, num_rhat_splits=2, transforms={'auto_loc': IdentityTransform(), 'auto_scale': biject_to(AutoDiagonalNormal.scale_constraint)}):
    """

    Returns (params, param_unconst_var, param_unconst_Rhat):
        - params: Parameters averaged over the last `burn_out` values in unconstrained optimisation space, then transformed into constrained parameter space.
        - param_unconst_var: Parameter variances in unconstrained optimisation space.
        - param_unconst_Rhat: split R-hat values of parameter traces in unconstrained optimisation space.
    """

    if isinstance(trace_tuple_or_dict, traces):
        trace_dict = traces_to_dict(trace_tuple_or_dict)
    elif isinstance(trace_tuple_or_dict, dict):
        trace_dict = trace_tuple_or_dict
    else: raise ValueError(f"trace_tuple_or_dict must be a traces tuple or a dict; was {type(trace_tuple_or_dict)}.")

    num_rhat_splits = np.minimum(num_rhat_splits, burn_out)

    params = {k: transforms[k](np.mean(transforms[k].inv(v[-burn_out:]), axis=0)) for k, v in trace_dict.items()}

    param_unconst_var = {k: np.var(v[-burn_out], axis=0) for k, v in trace_dict.items()}
    param_unconst_Rhat = split_Rhat(trace_dict, num_rhat_splits, only_last=burn_out, transforms=transforms)


    return params, param_unconst_var, param_unconst_Rhat

def average_converged_params(trace_tuple_or_dict: Union[traces, Dict[str, np.ndarray]], spacing=100, threshold=0.05, transforms={'auto_loc': IdentityTransform, 'auto_scale': biject_to(AutoDiagonalNormal.scale_constraint)}):
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

def load_params_converged(prefix, args, spacing=400, threshold=0.05, **kwargs):
    stored_model_name = filenamer(prefix, args, **kwargs)
    stored_model_path = f"{os.path.join(args.stored_model_dir, stored_model_name)}"

    traces_path = f"{stored_model_path}_traces.p"
    if os.path.exists(traces_path): # always read from traces file, if it exists
        with open(traces_path, "rb") as f:
            trace_tuple_or_dict = pickle.load(f) # type: traces
    else: # error if traces file does not exists and average over last epochs is requested
        raise ValueError(f"You asked to average over converged burn out, but no traces file is available at {traces_path}")

    return average_converged_params(trace_tuple_or_dict, spacing=spacing, threshld=threshold)

def load_params(prefix, args, num_rhat_splits=2, burn_out=None, **kwargs):
    """
    Loads parameters and averages in unconstrained space over the last `burn_out` (or `args.avg_over` if `burn_out` is None)
    epochs and compute their variance.

    Convenience wrapper for `average_params` that takes care of loading the file as well.

    Will always prioritize loading the corresponding traces pickle file, but falls back to loading the twinify
    output if no traces pickle file is available and `burn_out` (or `args.avg_over` if `burn_out` is None) equals 1.

    Returns (params, param_unconst_var, param_unconst_Rhat):
        - params: Parameters averaged in unconstrained optimisation space, then transformed into constrained parameter space.
        - param_unconst_var: Parameter variances in unconstrained optimisation space.
        - param_unconst_Rhat: split R-hat values of parameter traces in unconstrained optimisation space.
    """
    if burn_out is None:
        burn_out = args.avg_over
    stored_model_name = filenamer(prefix, args, **kwargs)
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

    return average_params(trace_tuple_or_dict, burn_out, num_rhat_splits)

from functools import reduce
import jax
def tree_reduce(reduce_fn, trees):
    return jax.tree_multimap(lambda *v: reduce(reduce_fn, v), *trees)


### Figure styles
default_aspect = 12. / 9.
neurips_fig_style = {'figure.figsize':(5.5, 5.5 / default_aspect), 'font.size':10}
