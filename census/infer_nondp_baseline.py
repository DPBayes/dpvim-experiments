"""
Inference script for testing the data generating model.
"""

import jax, pickle, argparse, tqdm, os

import jax.numpy as jnp
import numpy as np
import pandas as pd

from numpyro.optim import Adam
from numpyro.infer.svi import SVI
from numpyro.infer import Trace_ELBO

from d3p.minibatch import subsample_batchify_data,  q_to_batch_size
import d3p.random

from twinify.model_loading import load_custom_numpyro_model
from twinify.infer import InferenceException

from utils import traces


def initialize_rngs(seed):
    master_rng = d3p.random.PRNGKey(seed)
    print(f"RNG seed: {seed}")

    inference_rng, numpy_seed = d3p.random.split(master_rng, 2)

    numpy_seed = int(d3p.random.random_bits(numpy_seed, 32, (1,)))
    np.random.seed(numpy_seed)

    return inference_rng

from twinify.infer import _cast_data_tuple
def _train_model(rng, svi, data, batch_size, num_data, num_epochs, silent=False):
    rng, svi_rng, init_batch_rng = d3p.random.split(rng, 3)
    svi_rng = d3p.random.convert_to_jax_rng_key(svi_rng)

    assert(type(data) == tuple)
    data = _cast_data_tuple(data)
    init_batching, get_batch = subsample_batchify_data(data, batch_size)
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

    initial_params = svi.get_params(svi_state)
    locs_over_epochs = np.zeros((num_epochs + 1, initial_params['auto_loc'].shape[-1]))
    scales_over_epochs = np.zeros((num_epochs + 1, initial_params['auto_scale'].shape[-1]))
    locs_over_epochs[0] = initial_params['auto_loc']
    scales_over_epochs[0] = initial_params['auto_scale']

    rng, epochs_rng = d3p.random.split(rng)

    progressbar = tqdm.tqdm(range(num_epochs))
    for e in progressbar:
        batchify_rng = d3p.random.fold_in(epochs_rng, e)
        num_batches, batchify_state = init_batching(batchify_rng)

        svi_state, loss = train_epoch(num_batches, svi_state, batchify_state)
        params_after_epoch = svi.get_params(svi_state)
        locs_over_epochs[e + 1] = params_after_epoch['auto_loc']
        scales_over_epochs[e + 1] = params_after_epoch['auto_scale']
        if np.isnan(loss):
            raise InferenceException(
                traces(locs_over_epochs, scales_over_epochs)
            )
        loss /= num_data
        progressbar.set_description(f"epoch {e}: loss {loss}")

    return svi.get_params(svi_state), loss, \
        traces(locs_over_epochs, scales_over_epochs)


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
    parser.add_argument('data_path', type=str, help='Path to input data.')
    parser.add_argument('model_path', type=str, help='Path to model file (.txt or .py).')
    parser.add_argument("output_dir", type=str, help="Dir to store outputs.")
    parser.add_argument("--seed", default=None, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
    parser.add_argument("--k", default=50, type=int, help="Mixture components in fit (for automatic modelling only).")
    parser.add_argument("--num_epochs", "-e", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="Subsampling ratio for DP-SGD.")
    parser.add_argument("--save_traces", "-st", action='store_true', default=False, help="Save parameter traces.")

    args, unknown_args = parser.parse_known_args()

    # read the data
    try:
        df_whole = pd.read_csv(args.data_path, index_col=0)
    except Exception as e:
        print("#### UNABLE TO READ DATA FILE ####")
        print(e)
        return 1
    print("Loaded data set has {} rows (entries) and {} columns (features).".format(*df_whole.shape))


    train_df_whole = df_whole.copy()
    # remove the non-military service individuals
    train_df_whole = train_df_whole[(train_df_whole.iMilitary != 0) & (train_df_whole.iMilitary != 4)].copy()
    train_df_whole = train_df_whole.dropna()

    try:
        model, guide, preprocess_fn, postprocess_fn = load_custom_numpyro_model(args.model_path, args, unknown_args, train_df_whole)
    except (ModuleNotFoundError, FileNotFoundError) as e:
        print("#### COULD NOT FIND THE MODEL FILE ####")
        print(e)
        return 1

    train_data_whole, _, _ = preprocess_fn(train_df_whole)

    assert isinstance(train_data_whole, tuple)
    if len(train_data_whole) == 1:
        print("After preprocessing, the data has {} entries with {} features each.".format(*train_data_whole[0].shape))
    else:
        print("After preprocessing, the data was split into {} splits:".format(len(train_data_whole)))
        for i, x in enumerate(train_data_whole):
            print("\tSplit {} has {} entries with {} features each.".format(i, x.shape[0], 1 if x.ndim == 1 else x.shape[1]))

        # compute the needed dp scale for the dp-sgd
        train_data = train_data_whole
        print("Data contains {} entries with {} dimensions".format(*train_data[0].shape))
        num_data = len(train_data[0])

        # split rngs
        inference_rng = initialize_rngs(args.seed)

        # set up dpsvi algorithm of choice
        optimizer = Adam(1e-3)
        svi = SVI(
            model,
            guide,
            optimizer,
            Trace_ELBO(),
            num_obs_total=num_data
        )

        # train the model
        batch_size = q_to_batch_size(args.sampling_ratio, num_data)
        posterior_params, elbo, trace_tuple = _train_model(
                inference_rng,
                svi, train_data,
                batch_size, num_data, args.num_epochs
        )

        # save results
        from twinify.results import TwinifyRunResult
        from twinify import __version__
        result = TwinifyRunResult(
            posterior_params, elbo, args, unknown_args, __version__
        )

        from utils import filenamer
        output_name = filenamer("vanilla", "wholepop", args, epsilon="non_dp", clipping_threshold=None)
        output_path = f"{os.path.join(args.output_dir, output_name)}"
        print(f"Storing results to {output_path}")
        pickle.dump(result, open(f"{output_path}.p", "wb"))

        if args.save_traces:
            with open(f"{output_path}_traces.p", "wb") as f:
                pickle.dump(trace_tuple, f)

if __name__ == "__main__":
    main()
