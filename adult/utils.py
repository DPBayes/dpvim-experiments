import jax
import jax.numpy as jnp
import numpy as np

from d3p.minibatch import subsample_batchify_data
import d3p.random

import tqdm

def filenamer(prefix, args=None, **kwargs):
    def filenamer_explicit(prefix, epsilon=None, clipping_threshold=None, seed=None, num_epochs=None, **unused_kwargs):
        output_name=f"{prefix}_eps{epsilon}_C{clipping_threshold}_seed{seed}_epochs{num_epochs}"
        return output_name

    if isinstance(args, argparse.Namespace):
        new_kwargs = args.__dict__.copy()
        new_kwargs.update(kwargs)
        if 'prefix' in new_kwargs:
            del new_kwargs['prefix']
        kwargs = new_kwargs

    return filenamer_explicit(prefix, center, **kwargs)

def infer(dpsvi, px_grad_fn, data, batch_size, num_epochs, seed, auto_init_loc=False):
    rng_key = d3p.random.PRNGKey(seed)
    # set up minibatch sampling
    batchifier_init, get_batch = subsample_batchify_data(data, batch_size)
    _, batchifier_state = batchifier_init(rng_key)

    # set up DP-VI algorithm
    if type(data) is tuple:
        q = batch_size / len(data[0])
    num_iter_per_epoch = int(1./q)

    svi_state = dpsvi.init(rng_key, *get_batch(0, batchifier_state))

    if auto_init_loc==False:
        print("Manually initializing the auto_loc")
        # manually initialize the locs
        from d3p.svi import DPSVIState
        optim_params = dpsvi.optim.get_params(svi_state.optim_state)
        optim_params['auto_loc'] = 0.1*jax.random.normal(jax.random.PRNGKey(seed), shape=optim_params['auto_loc'].shape)
        optim_state = dpsvi.optim.init(optim_params)
        svi_state = DPSVIState(optim_state, svi_state.rng_key, observation_scale=1.)
        #

    per_example_grads_protos = px_grad_fn(svi_state, get_batch(0, batchifier_state))

    # run inference
    def step_function(i, val):
        svi_state, _ = val
        batch = get_batch(i, batchifier_state)

        per_example_grads = px_grad_fn(svi_state, batch)

        svi_state, _ = dpsvi.update(svi_state, *batch)

        return svi_state, per_example_grads

    per_example_grads_for_epochs = []
    params_for_epochs = []
    for e in tqdm.tqdm(range(num_epochs)):
        svi_state, per_example_grads = jax.lax.fori_loop(e * num_iter_per_epoch, (e + 1) * num_iter_per_epoch, step_function, (svi_state, per_example_grads_protos))
        per_example_grads_for_epochs.append(per_example_grads)
        params_for_epochs.append(dpsvi.get_params(svi_state))

    return params_for_epochs, per_example_grads_for_epochs

### Figure styles
default_aspect = 12. / 9.
neurips_fig_style = {'figure.figsize':(5.5, 5.5 / default_aspect), 'font.size':10}

from numpyro.infer.autoguide import AutoDiagonalNormal, biject_to
from numpyro.distributions.transforms import IdentityTransform
from typing import Optional, Dict, Iterable

def list_of_dicts_into_dict(trace: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {
        site: np.array([step_dict[site] for step_dict in trace]) for site in trace[0].keys()
    }

def dict_to_array(d):
    return np.array((d['auto_loc'], d['auto_scale']))

def array_to_dict(a):
    assert len(a) == 2
    return {
        'auto_loc': a[0],
        'auto_scale': a[1]
    }


def get_default_transforms():
    return {'auto_loc': IdentityTransform(), 'auto_scale': biject_to(AutoDiagonalNormal.scale_constraint)}

def single_param_site_Rhat(param_traces: np.ndarray, only_last: Optional[int] = None) -> float:
    """
    Args:
        param_traces: of shape [num_runs, num_iterations, ...]
    """
    assert param_traces.ndim >= 2
    if isinstance(only_last, int):
        param_traces = param_traces[:, -only_last:]

    # M = param_traces.shape[0]
    N = param_traces.shape[1]

    mean_within_run_variance = np.mean(np.var(param_traces, ddof=1, axis=1), axis=0)
    between_run_variance_div_N = np.var(np.mean(param_traces, axis=1), ddof=1, axis=0)
    marginal_posterior_variance = ((N - 1) / N) * mean_within_run_variance + between_run_variance_div_N

    Rhat = np.sqrt(marginal_posterior_variance / mean_within_run_variance)
    return Rhat

def multirun_Rhat(param_traces, only_last=None, transforms=get_default_transforms()):
    """
    Args:
        param_traces: Traces as (j)np arrays for each parameter site for each run. Maybe a dictionary
            containing numpy arrays of shape [num_runs, num_iterations, ...]
            or an iterable of length num_runs of dictionaries containing arrays of shape [num_iterations, ...].
        only_last (int): Optional. Only consider the last `only_last` iterations from each trace. If None, Rhat is computed
            over the entire given trace.
    """
    if not isinstance(param_traces, dict):
        reorganised_traces = list_of_dicts_into_dict(param_traces)
    else:
        reorganised_traces = param_traces

    return {
        site: single_param_site_Rhat(transforms[site].inv(values), only_last=only_last) for site, values in reorganised_traces.items()
    }

def split_Rhat(param_traces, num_splits=2, only_last=None, shuffle=False, transforms=get_default_transforms()):
    def split_Rhat_single_param_site(trace):
        assert trace.ndim >= 2
        if only_last is not None:
            trace = trace[-only_last:]
        if shuffle:
            trace = np.random.permutation(trace)
        trace_splits = np.array(np.split(trace, num_splits, axis=0))

        return single_param_site_Rhat(trace_splits)
    return {
        site: split_Rhat_single_param_site(transforms[site].inv(values)) for site, values in param_traces.items()
    }

from sklearn.linear_model import LinearRegression

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
    return fit.coef_.flatten(), fit.intercept_.flatten()

def single_linear_fit(param_traces: Dict[str, np.ndarray], only_last=None, transforms=get_default_transforms()):
    """
    Args:
        param_traces: Dictionary of parameter sites with values of shape [num_iterations, ...]
    """
    return {
        site: single_param_site_single_linear_fit(transforms[site].inv(values), only_last) for site, values in param_traces.items()
    }

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
    if np.any(np.isnan(param_trace)):
        coefs = np.nan * np.ones((len(np.arange(0, len(param_trace), spacing)), param_trace.shape[-1]))
        intercepts = np.nan * np.ones((len(np.arange(0, len(param_trace), spacing)), param_trace.shape[-1]))
        return coefs, intercepts
    for start_indx in np.arange(0, len(param_trace), spacing):
        fit = single_param_site_single_linear_fit(param_trace[start_indx:])
        coefs.append(fit[0])
        intercepts.append(fit[1])
        # x = np.expand_dims(np.linspace(0., 1., len(param_trace[start_indx:])), -1)
        # fit = LinearRegression().fit(x, param_trace[start_indx:])
        # coefs.append(fit.coef_.flatten())
        # intercepts.append(fit.intercept_.flatten())
    return np.array(coefs), np.array(intercepts)
    #return fit.coef_.flatten(), fit.intercept_


def linear_fit(param_traces: Dict[str, np.ndarray], only_last=None, spacing=None, transforms=get_default_transforms()):
    """
    Args:
        param_traces: Dictionary of parameter sites with values of shape [num_iterations, ...]
    """
    return {
        site: single_param_site_linear_fit(transforms[site].inv(values), only_last, spacing) for site, values in param_traces.items()
    }
