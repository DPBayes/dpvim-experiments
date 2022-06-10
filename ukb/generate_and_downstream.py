import jax, numpyro, d3p, pickle, argparse, tqdm, os, sys

import jax.numpy as jnp
import numpy as np
import pandas as pd
from collections import defaultdict

from twinify.model_loading import load_custom_numpyro_model
from twinify.infer import InferenceException
from twinify.sampling import sample_synthetic_data, reshape_and_postprocess_synthetic_data

from utils import filenamer, fit_model1, load_params, traces

def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
    parser.add_argument('data_path', type=str, help='Path to input data.')
    parser.add_argument('model_path', type=str, help='Path to model file (.txt or .py).')
    parser.add_argument("stored_model_dir", type=str, help="Dir from which to read learned parameters.")
    parser.add_argument("--output_dir", type=str, default=None, help="Dir to store the results")
    parser.add_argument("--prefix", type=str, help="Type of a DPSVI")
    parser.add_argument("--epsilon", type=str, help="Privacy level")
    def parse_clipping_threshold_arg(value): return None if value == "None" else float(value)
    parser.add_argument("--clipping_threshold", default=None, type=parse_clipping_threshold_arg, help="Clipping threshold for DP-SGD.")
    parser.add_argument("--seed", default=None, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
    parser.add_argument("--k", default=50, type=int, help="Mixture components in fit (for automatic modelling only).")
    parser.add_argument("--num_epochs", "-e", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--num_synthetic_data_sets", "--M", default=100, type=int, help="Number of synthetic data sets to apply the downstream")
    parser.add_argument("--avg_over", default=1, type=int, help="Model parameters are averaged over the last avg_over epochs in the parameter traces, to mitigate influence of gradient noise.")

    args, unknown_args = parser.parse_known_args()

    if args.output_dir is None:
        args.output_dir = args.stored_model_dir

    ########################################## Read original data
    # read the whole UKB data
    df_whole = pd.read_csv(args.data_path)
    print("Loaded data set has {} rows (entries) and {} columns (features).".format(*df_whole.shape))

    train_df_whole = df_whole.copy()
    train_df_whole = train_df_whole.dropna()

    model, _, preprocess_fn, postprocess_fn= load_custom_numpyro_model(args.model_path, args, unknown_args, train_df_whole)

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

    ########################################## Prepare reusable sampling function

    def sample_synthetic(rng, num_synthetic, model_params):
        num_parameter_samples = 1

        posterior_samples = sample_synthetic_data(
            model,
            guide,
            model_params,
            rng,
            num_parameter_samples,
            num_synthetic
        )

        # HACK: in very rare cases, numpyro <= 0.9.2 samples larger than number of categories from Categorical
        #       if probability vector does not sum exactly to 1 (due to numerics), which apparently happens
        #       sometimes in AutoDiagonalNormal; fix by clipping those to last category
        posterior_samples['X'] = np.clip(
            posterior_samples['X'], 0, np.max(train_data_whole[0], axis=0)
        )

        # post-process
        encoded_syn_df = reshape_and_postprocess_synthetic_data(
            posterior_samples, conditioned_postprocess_fn, True, num_parameter_samples
        )
        return next(encoded_syn_df)[1]

    ##########################################


    sampling_rngs = jax.random.split(jax.random.PRNGKey(args.seed), args.num_synthetic_data_sets)
    downstream_results = []

    # read posterior params from file
    model_params, _, params_Rhat = load_params(args.prefix, args)

    # sample synthetic data
    num_synthetic = len(train_df_whole["assessment_center"])

    for sampling_idx, sampling_rng in enumerate(sampling_rngs):
        print(f"#### PROCESSING SYNTHETIC DATA SET {sampling_idx+1} / {args.num_synthetic_data_sets}")

        # sample synthetic data
        print("  Sampling")
        encoded_syn_df = sample_synthetic(sampling_rng, num_synthetic, model_params)

        # downstream
        print("  Doing downstream inference")
        statsmodels_fit, _ = fit_model1(encoded_syn_df)
        downstream_results.append(statsmodels_fit.summary())


    ## store results
    avg_prefix = "" if args.avg_over == 1 else f"avg{args.avg_over}_"
    wholepop_output_name = "downstream_results_" + avg_prefix + filenamer(args.prefix, args)
    wholepop_output_path = f"{os.path.join(args.output_dir, wholepop_output_name)}.p"
    with open(wholepop_output_path, "wb") as f:
        pickle.dump(downstream_results, f)

    rhat_output_path = f"{os.path.join(args.output_dir, wholepop_output_name)}_rhat.p"
    with open(rhat_output_path, "wb") as f:
        pickle.dump(params_Rhat, f)


if __name__ == "__main__":
    main()
