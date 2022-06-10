import jax, numpyro, d3p, pickle, argparse, tqdm, os, sys

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
    parser.add_argument("--k", default=16, type=int, help="Mixture components in fit (for automatic modelling only).")
    parser.add_argument("--num_epochs", "-e", default=1000, type=int, help="Number of training epochs.")
    parser.add_argument("--num_synthetic_data_sets", "--M", default=100, type=int, help="Number of synthetic data sets to apply the downstream")
    parser.add_argument("--avg_over", default=1, type=int, help="Model parameters are averaged over the last avg_over epochs in the parameter traces, to mitigate influence of gradient noise.")

    args, unknown_args = parser.parse_known_args()

    if args.output_dir is None:
        args.output_dir = args.stored_model_dir

    # read posterior params from file
    _, _, params_Rhat = load_params(args.prefix, args)

    avg_prefix = "" if args.avg_over == 1 else f"avg{args.avg_over}_"
    wholepop_output_name = "downstream_results_" + avg_prefix + filenamer(args.prefix, args)
    rhat_output_path = f"{os.path.join(args.output_dir, wholepop_output_name)}_rhat.p"
    with open(rhat_output_path, "wb") as f:
        pickle.dump(params_Rhat, f)


if __name__ == "__main__":
    main()
