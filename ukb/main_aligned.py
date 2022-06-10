import jax, numpyro, d3p, pickle, argparse, tqdm, os

import jax.numpy as jnp
import numpy as np
import pandas as pd

from numpyro.optim import Adam

from d3p.minibatch import subsample_batchify_data,  q_to_batch_size
from d3p.dputil import approximate_sigma_remove_relation

from twinify.model_loading import load_custom_numpyro_model
from twinify.infer import InferenceException
from twinify.sampling import sample_synthetic_data, reshape_and_postprocess_synthetic_data

import d3p.random
import chacha.defs
import secrets

from collections import namedtuple
traces = namedtuple('traces', ['loc_trace', 'scale_trace'])

from dpsvi import AlignedGradientDPSVI

"""
Inference script for testing the data generating model with the aligned DPVI
"""

from twinify import __version__


def initialize_rngs(seed):
    if seed is None:
        seed = secrets.randbits(chacha.defs.ChaChaKeySizeInBits)
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
    from twinify.infer import _cast_data_tuple
    data = _cast_data_tuple(data)
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

    progressbar = tqdm.tqdm(range(num_epochs))
    for e in progressbar:
        batchify_rng = rng_suite.fold_in(epochs_rng, e)
        num_batches, batchify_state = init_batching(batchify_rng)

        svi_state, loss = train_epoch(num_batches, svi_state, batchify_state)
        params_after_epoch = svi.get_params(svi_state)
        if e == 0:
            locs_over_epochs = np.zeros((num_epochs, params_after_epoch['auto_loc'].shape[-1]))
            scales_over_epochs = np.zeros((num_epochs, params_after_epoch['auto_scale'].shape[-1]))
        locs_over_epochs[e] = params_after_epoch['auto_loc']
        scales_over_epochs[e] = params_after_epoch['auto_scale']
        if np.isnan(loss):
            raise InferenceException
        loss /= num_data
        progressbar.set_description(f"epoch {e}: loss {loss}")

    return svi.get_params(svi_state), loss, (locs_over_epochs, scales_over_epochs)


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
    parser.add_argument('data_path', type=str, help='Path to input data.')
    parser.add_argument('model_path', type=str, help='Path to model file (.txt or .py).')
    parser.add_argument("output_dir", type=str, help="Dir to store outputs.")
    parser.add_argument("--epsilon", type=str, help="Privacy level")
    def parse_clipping_threshold_arg(value): return None if value == "None" else float(value)
    parser.add_argument("--clipping_threshold", default=None, type=parse_clipping_threshold_arg, help="Clipping threshold for DP-SGD.")
    parser.add_argument("--seed", default=None, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
    parser.add_argument("--k", default=50, type=int, help="Mixture components in fit (for automatic modelling only).")
    parser.add_argument("--num_epochs", "-e", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="Subsampling ratio for DP-SGD.")
    parser.add_argument("--save_traces", "-st", action='store_true', default=False, help="Save parameter traces.")
    parser.add_argument("--sample_synthetic", action='store_true', default=False, help="Sample and store a synthetic data set. If not set, only a inferred parameters are saved.")
    parser.add_argument("--num_synthetic", "--n", default=None, type=int, help="Amount of synthetic data to generate in total. By default as many as input data.")
    parser.add_argument("--num_synthetic_records_per_parameter_sample", "--m", default=1, type=int, help="Amount of synthetic samples to sample per parameter value drawn from the learned parameter posterior.")
    parser.add_argument("--separate_output", default=False, action='store_true', help="Store synthetic data in separate files per parameter sample.")

    args, unknown_args = parser.parse_known_args()

    # read the whole UKB data
    try:
        df_whole = pd.read_csv(args.data_path)
    except Exception as e:
        print("#### UNABLE TO READ DATA FILE ####")
        print(e)
        return 1
    print("Loaded data set has {} rows (entries) and {} columns (features).".format(*df_whole.shape))

    train_df_whole = df_whole.copy()
    train_df_whole = train_df_whole.dropna()

    try:
        model, guide, preprocess_fn, postprocess_fn = load_custom_numpyro_model(args.model_path, args, unknown_args, train_df_whole)
    except (ModuleNotFoundError, FileNotFoundError) as e:
        print("#### COULD NOT FIND THE MODEL FILE ####")
        print(e)
        return 1

    train_data_whole, num_data_whole, feature_names = preprocess_fn(train_df_whole)

    #
    assert isinstance(train_data_whole, tuple)
    if len(train_data_whole) == 1:
        print("After preprocessing, the data has {} entries with {} features each.".format(*train_data_whole[0].shape))
    else:
        print("After preprocessing, the data was split into {} splits:".format(len(train_data_whole)))
        for i, x in enumerate(train_data_whole):
            print("\tSplit {} has {} entries with {} features each.".format(i, x.shape[0], 1 if x.ndim == 1 else x.shape[1]))

    # compute the needed dp scale for the dp-sgd
    train_data = train_data_whole
    print("Whole population contains {} entries with {} dimensions".format(*train_data[0].shape))
    num_data = len(train_data[0])

    if args.epsilon != 'non_dp':
        num_total_iters = np.ceil(args.num_epochs / args.sampling_ratio)
        dp_scale, epsilon, _ = approximate_sigma_remove_relation(
                float(args.epsilon),
                1./num_data,
                args.sampling_ratio,
                num_total_iters
        )
    else:
        dp_scale = 0.0
    print(f"Adding noise with std. {dp_scale}")

    # split rngs
    inference_rng, sampling_rng = initialize_rngs(args.seed)

    # re-read the model and guide to make sure that there is no funny business
    model, guide, _, _ = load_custom_numpyro_model(args.model_path, args, unknown_args, train_df_whole)

    # train model
    optimizer = Adam(1e-3)
    svi = AlignedGradientDPSVI(
        model,
        guide,
        optimizer,
        args.clipping_threshold,
        dp_scale,
        num_obs_total=num_data
    )

    batch_size = q_to_batch_size(args.sampling_ratio, num_data)
    posterior_params, elbo, (locs_over_epochs, scales_over_epochs)= _train_model(
            inference_rng, d3p.random,
            svi, train_data,
            batch_size, num_data, args.num_epochs
            )

    # save results
    from twinify.__main__ import TwinifyRunResult
    result = TwinifyRunResult(
        posterior_params, elbo, args, unknown_args, __version__
    )
    from utils import filenamer
    output_name = filenamer("aligned", args)
    output_path = f"{os.path.join(args.output_dir, output_name)}"
    print(f"Storing results to {output_path}")
    pickle.dump(result, open(f"{output_path}.p", "wb"))

    if args.save_traces:
        trace_tuple = traces(
            locs_over_epochs,
            scales_over_epochs
        )
        pickle.dump(trace_tuple, open(f"{output_path}_traces.p", "wb"))

    # sample synthetic data
    if args.sample_synthetic:
        print("Model learning complete; now sampling data!")
        num_synthetic = args.num_synthetic
        if num_synthetic is None:
            num_synthetic = num_data

        num_parameter_samples = int(np.ceil(num_synthetic / args.num_synthetic_records_per_parameter_sample))
        num_synthetic = num_parameter_samples * args.num_synthetic_records_per_parameter_sample
        print(f"Will sample {args.num_synthetic_records_per_parameter_sample} synthetic data records for each of "
            f"{num_parameter_samples} samples from the parameter posterior for a total of {num_synthetic} records.")
        if args.separate_output:
            print("They will be stored in separate data sets for each parameter posterior sample.")
        else:
            print("They will be stored in a single large data set.")
        posterior_samples = sample_synthetic_data(
            model, guide, posterior_params, sampling_rng, num_parameter_samples, args.num_synthetic_records_per_parameter_sample
        )

        # postprocess: so that the synthetic twin looks like the original data
        #   - extract samples from the posterior_samples dictionary and construct pd.DataFrame
        #   - if preprocessing involved data mapping, it is mapped back here
        conditioned_postprocess_fn = lambda posterior_samples: postprocess_fn(posterior_samples, df_whole, feature_names)
        for i, (syn_df, encoded_syn_df) in enumerate(reshape_and_postprocess_synthetic_data(
            posterior_samples, conditioned_postprocess_fn, args.separate_output, num_parameter_samples
        )):
            if args.separate_output:
                filename = f"{output_path}.{i}.csv"
            else:
                filename = f"{output_path}.csv"
            encoded_syn_df.to_csv(filename, index=False)

if __name__ == "__main__":
    main()
