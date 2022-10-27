import jax, numpyro, os
import numpy as np
import jax.numpy as jnp
import pandas as pd

from numpyro.primitives import sample, plate
from numpyro.distributions import Normal, Gamma, MultivariateNormal, LKJCholesky

def model(X=None, y=None, num_obs_total=None):
    batch_size = 1

    if y is not None:
        batch_size = y.shape[0]

    if X is not None:
        batch_size, d = X.shape

    if num_obs_total is None:
        num_obs_total = batch_size

    ############## PRIORS ##################
    weight = sample("weight", MultivariateNormal(jnp.zeros(d), jnp.eye(d)))

    std = sample("noise_std", Gamma(0.1, 0.1))

    ############# LIKELIHOOD ####################

    with plate('batch', num_obs_total, batch_size):

        mean = X @ weight
        y = sample('y', Normal(mean, std), obs=y)

def generate_corr_chol_from_LKJ(rs, d_without_intercept, correlation_strength):
    stds = np.exp(0.2 * rs.randn(d_without_intercept))

    # NOTE this currently ignores random state and simply reseeds jax rng. ideally we would derive that from the given rs.
    # but in practice we keep that one constant anyways, so this does no harm here.
    L = LKJCholesky(d_without_intercept, concentration=correlation_strength).sample(jax.random.PRNGKey(123))
    return np.diag(stds) @ L

def generate_corr_chol_manual(rs, d_without_intercept, density, correlation_strength):
    stds = np.exp(0.2 * rs.randn(d_without_intercept))

    tril_idxs = np.tril_indices(d_without_intercept, k=-1)
    num = len(tril_idxs[0])

    # choice_rng, beta_rng, flip_rng = jax.random.split(rng, 3)
    idxs = rs.choice(num, int(density * num), replace=False)
    # idxs = jax.random.choice(choice_rng, jnp.arange(num), shape=(int(density * num),), replace=False)
    tril_idxs = tuple(x[idxs] for x in tril_idxs)

    bs = rs.beta(10. * correlation_strength * np.ones_like(idxs), 10 * np.ones_like(idxs))
    fs = (rs.rand(len(idxs)) >= .5) * 2 - 1

    cs = bs * fs

    L = np.eye(d_without_intercept)
    L[tril_idxs] = cs

    cov_diag_entries = (L**2).sum(axis=1)

    normalising_factor = 1. / np.sqrt(cov_diag_entries)

    # L = L.at[tril_idxs].set(cs)
    return np.diag(stds) @ L @ np.diag(normalising_factor)

def generate_data(rs, L, n_train, noise_std=1., true_weight=None):
    d_without_intercept = L.shape[0]

    # generate X
    X = rs.randn(n_train, d_without_intercept) @ L.T

    # add intercept
    X = np.pad(X, ((0, 0), (1, 0)), mode="constant", constant_values=1.0)

    # generate y
    if true_weight is None:
        true_weight = rs.randn(d_without_intercept+1)
    y = X @ true_weight + noise_std * rs.randn(n_train)

    return jnp.array(X), jnp.array(y), true_weight
