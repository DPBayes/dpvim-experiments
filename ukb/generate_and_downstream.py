"""
This script samples synthetic data sets from a generative model learned using `infer.py`
and performs the downstream analysis task (Poisson regression following Niedzwiedz et al.).
"""

import jax, numpyro, d3p, pickle, argparse, tqdm, os

import jax.numpy as jnp
import numpy as np
import pandas as pd
from collections import defaultdict

from twinify.model_loading import load_custom_numpyro_model

from utils import filenamer, fit_model1, load_params, traces, format_statsmodels_summary_as_df, load_params_converged

def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
    parser.add_argument('data_path', type=str, help='Path to input data.')
    parser.add_argument('model_path', type=str, help='Path to model python file.')
    parser.add_argument("stored_model_dir", type=str, help="Dir from which to read learned parameters.")
    parser.add_argument("--output_dir", type=str, default=None, help="Dir to store the results")
    parser.add_argument("--epsilon", type=str, help="Privacy level")
    parser.add_argument("--clipping_threshold", default=2.0, type=float, help="Clipping threshold for DP-SGD.")
    parser.add_argument("--seed", default=None, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
    parser.add_argument("--k", default=50, type=int, help="Mixture components in fit (for automatic modelling only).")
    parser.add_argument("--num_epochs", "-e", default=200, type=int, help="Number of training epochs.")
    parser.add_argument("--num_synthetic_data_sets", "--M", default=100, type=int, help="Number of synthetic data sets to apply the downstream")
    parser.add_argument("--avg_over", default=1, type=str, help="Model parameters are averaged over the last avg_over epochs in the parameter traces, to mitigate influence of gradient noise. Integer or 'converged'.")
    parser.add_argument("--convergence_test_threshold", default=0.05, type=float)
    parser.add_argument("--convergence_test_spacing", default=100, type=int)
    parser.add_argument("--dpvi_flavour", default="aligned", choices=['aligned', 'ng', 'aligned_ng', 'precon', 'vanilla'])

    args, unknown_args = parser.parse_known_args()

    if args.output_dir is None:
        args.output_dir = args.stored_model_dir

    if args.avg_over == "converged":
        avg_prefix = "avgconverged_"
    else:
        args.avg_over = int(args.avg_over)
        avg_prefix = "" if args.avg_over == 1 else f"avg{args.avg_over}_"

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
    ## NOTE this way of sampling seemed  to be much faster than the commented variant below.
    ## BUT, this is conditional on always sampling the same number of synthetic draws!! I guess this
    ## prevents triggering the recompilation in jax...
    ## We can still use it, by just sampling more than we need and throw away the 'excess' samples
    from numpyro.infer import Predictive
    @jax.jit
    def jitted_ppd_sampler(model_params, posterior_rng, sample_rng):

        # sample single parameter vector
        posterior_sampler = Predictive(
            guide, params=model_params, num_samples=1
        )

        posterior_samples = posterior_sampler(posterior_rng)
        # models always add a superfluous batch dimensions, squeeze it
        posterior_samples = {k: v.squeeze(0) for k,v in posterior_samples.items()}

        # sample num_record_samples_per_parameter_sample data samples
        ppd_sampler = Predictive(model, posterior_samples, batch_ndims=0)

        ppd_sample = ppd_sampler(sample_rng)
        return ppd_sample

    def sample_synthetic(rng, num_synthetic, model_params):
        posterior_rng, ppd_rng = jax.random.split(rng)
        #
        per_sample_rngs = jax.random.split(ppd_rng, num_synthetic)

        fixed_sampler = lambda sample_rng: jitted_ppd_sampler(model_params, posterior_rng, sample_rng)
        posterior_samples = jax.vmap(fixed_sampler)(per_sample_rngs)

        posterior_samples['X'] = np.clip(
            posterior_samples['X'], 0, np.max(train_data_whole[0], axis=0)
        )
        # models always add a superfluous batch dimensions, squeeze it
        squeezed_posterior_samples = {k: v.squeeze(1) for k, v in posterior_samples.items()}
        return conditioned_postprocess_fn(squeezed_posterior_samples)[1]

    ##########################################

    # read posterior params from file
    if args.avg_over == 'converged':
        model_params = load_params_converged(args.dpvi_flavour, "", args, threshold=args.convergence_test_threshold, spacing=args.convergence_test_spacing)
    else:
        model_params, _ = load_params(args.dpvi_flavour, "", args)

    # sample synthetic data
    num_synthetic = num_data_whole

    sampling_rngs = jax.random.split(jax.random.PRNGKey(args.seed), args.num_synthetic_data_sets)
    downstream_results = []

    for sampling_idx, sampling_rng in enumerate(sampling_rngs):
        print(f"#### PROCESSING SYNTHETIC DATA SET {sampling_idx+1} / {args.num_synthetic_data_sets}")

        # sample synthetic data
        print("  Sampling")
        syn_df = sample_synthetic(sampling_rng, num_synthetic, model_params)

        # downstream
        print("  Doing downstream inference")
        try:
            statsmodels_fit, _ = fit_model1(syn_df)
            summary_as_df = format_statsmodels_summary_as_df(statsmodels_fit.summary())
            downstream_results.append(summary_as_df)
        except:
            pass

    # if conversion of statsmodel result to dataframe failed, fill downstream results with nans until expected length
    downstream_results += [(np.nan * downstream_results[-1])] * (args.num_synthetic_data_sets - len(downstream_results))


    ## store results
    output_name = "downstream_results_" + avg_prefix + filenamer(args.dpvi_flavour, "", args)
    output_path = f"{os.path.join(args.output_dir, output_name)}.p"
    with open(output_path, "wb") as f:
        pickle.dump(downstream_results, f)


if __name__ == "__main__":
    main()
