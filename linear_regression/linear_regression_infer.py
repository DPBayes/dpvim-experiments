"""
This script runs the inference for Bayesian linear regression using the different DPVI
variants on simulated data with a given correlation strength and dimensionality.
"""

import jax, numpyro, tqdm, os
import numpy as np
import jax.numpy as jnp
import pandas as pd

from numpyro.infer import SVI, Trace_ELBO

from twinify.infer import InferenceException

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--clipping_threshold", type=float, default=2.0)
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--num_dims", type=int, default=100)
parser.add_argument("--results_path", type=str, default="./results/")
parser.add_argument("--corr_strength", type=float, default=1.0)
parser.add_argument("--run_aligned_full", default=False, action='store_true')
parser.add_argument("--run_vanilla_full", default=False, action='store_true')
parser.add_argument("--corr_method", choices=['LKJ', 'manual'], default="LKJ")
args = parser.parse_args()


from common import model, generate_data, generate_corr_chol_from_LKJ, generate_corr_chol_manual

from numpyro.infer.autoguide import AutoMultivariateNormal
guide = AutoMultivariateNormal(model)

### Generate data
n_train = 10000
d_without_intercept = args.num_dims
np_rng = np.random.RandomState(seed=123)
if args.corr_method == "LKJ":
    L = generate_corr_chol_from_LKJ(np_rng, d_without_intercept, args.corr_strength)
else:
    L = generate_corr_chol_manual(np_rng, d_without_intercept, args.corr_strength, .8)
X_train, y_train, true_weight = generate_data(np_rng, L, n_train)
print("max abs offdiag elem from covariance matrix: ", np.abs(np.tril(L @ L.T, -1)).max())

######## setup the training pipeline
from d3p.minibatch import subsample_batchify_data,  q_to_batch_size
import d3p

def initialize_rngs(seed):
    master_rng = d3p.random.PRNGKey(seed)
    print(f"RNG seed: {seed}")

    inference_rng, sampling_rng, numpyro_seed = d3p.random.split(master_rng, 3)
    sampling_rng = d3p.random.convert_to_jax_rng_key(sampling_rng)

    numpyro_seed = int(d3p.random.random_bits(numpyro_seed, 32, (1,)))
    np.random.seed(numpyro_seed)

    return inference_rng, sampling_rng

def _train_model(rng, rng_suite, svi, data, batch_size, num_data, num_epochs, silent=False):
    rng, svi_rng, init_batch_rng = rng_suite.split(rng, 3)

    assert(type(data) == tuple)
    init_batching, get_batch = subsample_batchify_data(data, batch_size, rng_suite=rng_suite)
    _, batchify_state = init_batching(init_batch_rng)

    batch = get_batch(0, batchify_state)
    svi_state = svi.init(svi_rng, *batch)

    @jax.jit
    def train_epoch(num_iters_for_epoch, svi_state, batchify_state):
        def update_iteration(i, state_and_loss):
            svi_state, loss = state_and_loss
            batch = get_batch(i, batchify_state)
            svi_state, iter_loss = svi.update(svi_state, *batch)
            return (svi_state, loss + iter_loss / num_iters_for_epoch)

        return jax.lax.fori_loop(0, num_iters_for_epoch, update_iteration, (svi_state, 0.))

    rng, epochs_rng = rng_suite.split(rng)

    progressbar = range(num_epochs) #tqdm.tqdm(range(num_epochs))
    #progressbar = tqdm.tqdm(range(num_epochs))
    for e in progressbar:
        batchify_rng = rng_suite.fold_in(epochs_rng, e)
        num_batches, batchify_state = init_batching(batchify_rng)

        svi_state, loss = train_epoch(num_batches, svi_state, batchify_state)
        params_after_epoch = svi.get_params(svi_state)
        if e == 0:
            locs_over_epochs = np.zeros((num_epochs, params_after_epoch['auto_loc'].shape[-1]))
        locs_over_epochs[e] = params_after_epoch['auto_loc']
        if np.isnan(loss):
            raise InferenceException()
        loss /= num_data
        #progressbar.set_description(f"epoch {e}: loss {loss}")

    return svi.get_params(svi_state), loss, (locs_over_epochs,)

def main():
    #
    if args.clipping_threshold == "None":
        clipping_threshold = None
    else:
        clipping_threshold = float(args.clipping_threshold)

    eps = args.epsilon
    sampling_ratio = 0.01
    train_data = (X_train, y_train)
    num_epochs = args.num_epochs
    batch_size = q_to_batch_size(sampling_ratio, n_train)

    from d3p.dputil import approximate_sigma
    num_iters = num_epochs * int(1/sampling_ratio)
    dp_scale = approximate_sigma(eps, 0.1/n_train, sampling_ratio, num_iters)[0]


    import pickle
    import os
    def store_result(file_name, data):
        if os.path.exists(file_name):
            os.replace(file_name, file_name + ".backup")
        with open(file_name, "wb") as f:
            pickle.dump(data, f)

    from fullrank_dpsvi import AlignedGradientFullRankDPSVI, VanillaFullRankDPSVI

    seed = args.seed
    filename = lambda variant: f"linreg_params_corr{args.corr_strength}_ndim{d_without_intercept}_{variant}_eps{eps}_clip{clipping_threshold}_seed{seed}.p"

    ####### aligned
    if args.run_aligned_full:
        svi_aligned = AlignedGradientFullRankDPSVI(
            model,
            guide,
            numpyro.optim.Adam(1e-3),
            clipping_threshold,
            dp_scale,
            num_obs_total=n_train
        )

        inference_rng, _ = initialize_rngs(seed)

        posterior_params_aligned, elbo_aligned, (locs_over_epochs_aligned,)= _train_model(
            inference_rng, d3p.random,
            svi_aligned, train_data,
            batch_size, n_train, num_epochs
        )
        print(f"final loss aligned: {elbo_aligned}")

        store_result(
            os.path.join(args.results_path, filename('aligned_fullrank')),
            posterior_params_aligned
        )

    ###### vanilla
    if args.run_vanilla_full:
        svi_vanilla = VanillaFullRankDPSVI(
            model,
            guide,
            numpyro.optim.Adam(1e-3),
            clipping_threshold,
            dp_scale,
            num_obs_total=n_train
        )

        inference_rng, _ = initialize_rngs(seed)

        posterior_params_vanilla, elbo_vanilla, (locs_over_epochs_vanilla,)= _train_model(
            inference_rng, d3p.random,
            svi_vanilla, train_data,
            batch_size, n_train, num_epochs
        )
        print(f"final loss vanilla: {elbo_vanilla}")

        store_result(
            os.path.join(args.results_path, filename('vanilla_fullrank')),
            posterior_params_vanilla
        )

    ############
    # non-dp

    nondp_path = os.path.join(args.results_path, filename('nondp_fullrank'))
    if not os.path.exists(nondp_path):
        print("running non dp inference (full rank)")

        optimizer = numpyro.optim.Adam(step_size=1e-3)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(jax.random.PRNGKey(seed), num_iters, X_train, y_train, n_train)

        ### save results
        store_result(nondp_path, svi_result.params)

if __name__ == "__main__":
    main()
