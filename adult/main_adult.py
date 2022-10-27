import jax, pickle, argparse, os, sys

import pandas as pd
import numpy as np

import jax.numpy as jnp

from utils import infer
from dpsvi import AlignedGradientDPSVI, AlignedNaturalGradientDPSVI, NaturalGradientDPSVI, PreconditionedGradientDPSVI, VanillaDPSVI

parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
parser.add_argument("variant", choices=('all', 'ng', 'vanilla', 'aligned', 'aligned_ng', 'precon', 'nondp'), help="Which variant of DPSVI to run")
parser.add_argument("--epsilon", type=float, default=None, help="Privacy level")
parser.add_argument("--dp_scale", type=float, default=None, help="DP-SGD noise std")
parser.add_argument("--seed", default=123, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
parser.add_argument("--num_epochs", "-e", default=4000, type=int, help="Number of training epochs.")
parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="Subsampling ratio for DP-SGD.")
# parser.add_argument("--store_gradients", action="store_true", default=False)
parser.add_argument("--output_path", type=str, default="./results/", help="Path to folder in which to store parameter traces; defaults to '../results'")
parser.add_argument("--init_auto_scale", type=float, default=0.1, help="Initial std for the VI posterior")
parser.add_argument("--optim", type=str, default="adam", help="Gradient optimizer")
parser.add_argument("--sgd_lr", type=float, default=1e-6, help="Learning rate for the SGD optimizer")


## model
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.primitives import sample, plate, param
from numpyro.distributions import Normal, Bernoulli, constraints

def model(xs, ys, N):
    batch_size, d = xs.shape
    w = sample('w', Normal(0, 1.), sample_shape=(d,))
    with plate('batch', N, batch_size):
        logit = xs.dot(w)
        sample('ys', Bernoulli(logits=logit), obs=ys)

def add_intercept(X):
    return jnp.pad(X, ((0,0), (0, 1)), constant_values=1)


def main():
    args, _ = parser.parse_known_args()
    print(args)

    ## check that output path exists
    if not os.path.exists(args.output_path):
        print("Output path does not exists, exiting!")
        sys.exit()

    ## load data

    from load_adult import preprocess_adult

    preprocessed_train_data, preprocessed_test_data, _, data_description = preprocess_adult()

    X_train = preprocessed_train_data.copy()
    y_train = X_train.pop("income")

    X_test = preprocessed_test_data.copy()
    y_test = X_test.pop("income")

    ## adjust the categorical variables with the largest group
    for key, value in data_description.items():
        if value != "continuous" and key != "income":
            categorical_names = [name for name in preprocessed_train_data.columns if key in name]
            largest_category_indx = preprocessed_train_data[categorical_names].sum().argmax()
            largest_category = categorical_names[largest_category_indx]
            del X_train[largest_category]
            del X_test[largest_category]


    ## add intercept
    X_train = add_intercept(X_train.values)
    y_train = jnp.array(y_train.values)

    X_test = add_intercept(X_test.values)
    y_test = y_test.values

    guide = AutoDiagonalNormal(model, init_scale=args.init_auto_scale)

    ## infer
    from numpyro.optim import Adam, SGD
    if args.optim == "adam":
        optimiser = Adam(1e-3)

    elif args.optim == "sgd":
        optimiser = SGD(args.sgd_lr)

    num_epochs = args.num_epochs

    C_ng = 0.1 # chosen by hand
    C_vanilla = 3.0 # chosen by hand
    C_aligned = 3.0 # chosen by hand
    C_aligned_ng = 0.1 # chosen by hand
    C_precon = 4.0 # chosen by hand

    q = args.sampling_ratio
    N_train = len(X_train)

    # compute the dp_scale
    from d3p.dputil import approximate_sigma
    if args.epsilon is not None:
        dp_scale = approximate_sigma(args.epsilon, 1/N_train, q, num_epochs / q)[0]
    else:
        assert(args.dp_scale is not None)
        dp_scale = args.dp_scale

    if args.epsilon is None and args.dp_scale is not None:
        dp_name_part = f"dp_scale{dp_scale}"
    elif args.epsilon is not None:
        dp_name_part = f"eps{args.epsilon}"

    ## filename template
    def filenamer(variant, suffix, C):
        if args.optim == "adam":
            return f"{args.output_path}/adult_{variant}_ne{num_epochs}_C{C}_q{q}_{dp_name_part}_auto_scale{args.init_auto_scale}_seed{seed}_optim{args.optim}_{suffix}"
        elif args.optim == "sgd":
            return f"{args.output_path}/adult_{variant}_ne{num_epochs}_C{C}_q{q}_{dp_name_part}_auto_scale{args.init_auto_scale}_seed{seed}_optim{args.optim}_lr{args.sgd_lr}_{suffix}"
    ##

    # naturalgrad
    naturalgrad_dpsvi = NaturalGradientDPSVI(model, guide, optimiser, C_ng, dp_scale, N=N_train)
    def naturalgrad_grad_fn(state, batch):

        ## compute vanilla gradients
        per_example_grads = naturalgrad_dpsvi._compute_per_example_gradients(state, *batch)[2]

        ## compute natural gradients
        params = naturalgrad_dpsvi.optim.get_params(state.optim_state)
        constrained_params = naturalgrad_dpsvi.constrain_fn(params)
        scales = constrained_params['auto_scale']

        scale_transform_d = jax.vmap(jax.grad(lambda x: naturalgrad_dpsvi.constrain_fn(x)['auto_scale']))(params)['auto_scale']

        ## scale to natural gradient (assuming autodiagonalguide)
        per_example_grads['auto_loc'] = scales**2 * per_example_grads['auto_loc']
        per_example_grads['auto_scale'] = .5 * (scales / scale_transform_d)**2 * per_example_grads['auto_scale']

        return per_example_grads

    # pre-conditioned dpsvi
    precon_dpsvi = PreconditionedGradientDPSVI(model, guide, optimiser, C_precon, dp_scale, N=N_train)
    def preconditioned_grad_fn(svi_state, batch):
        per_example_grads = precon_dpsvi._compute_per_example_gradients(svi_state, *batch)[2]

        ## compute natural gradients
        params = precon_dpsvi.optim.get_params(svi_state.optim_state)
        scale_transform_d = jax.vmap(jax.grad(lambda x: precon_dpsvi.constrain_fn(x)['auto_scale']))(params)['auto_scale']

        ## scale auto_scale gradients (back) up to same magnitude as auto_loc (because: g_{auto_scale} \approx T'(s) * g_{auto_loc})
        per_example_grads['auto_scale'] = (1./scale_transform_d) * per_example_grads['auto_scale']
        return per_example_grads

    ####### Train over seeds
    seed = args.seed
    # ng
    if args.variant == "ng" or args.variant == "all":
        print("natural gradient")
        naturalgrad_dpsvi_params_for_epochs, naturalgrad_dpsvi_grads_per_epochs = infer(
                naturalgrad_dpsvi,
                naturalgrad_grad_fn,
                (X_train, y_train),
                int(q * N_train),
                num_epochs,
                seed
        )
        ### store results
        filename = filenamer("ng", "trace", C_ng) + ".p"
        pd.to_pickle(naturalgrad_dpsvi_params_for_epochs, filename)

    # aligned natural
    aligned_ng_dpsvi = AlignedNaturalGradientDPSVI(model, guide, optimiser, C_aligned_ng, dp_scale, N=N_train)
    if args.variant == "aligned_ng" or args.variant == "all":
        print("aligned natural gradient")
        aligned_ng_dpsvi_params_for_epochs, aligned_ng_dpsvi_grads_per_epochs = infer(
                aligned_ng_dpsvi,
                lambda state, batch: aligned_ng_dpsvi._compute_per_example_gradients(state, *batch)[2],
                (X_train, y_train),
                int(q * N_train),
                num_epochs,
                seed
        )
        filename = filenamer("aligned_ng", "trace", C_aligned_ng) + ".p"
        pd.to_pickle(aligned_ng_dpsvi_params_for_epochs, filename)

    # aligned
    aligned_dpsvi = AlignedGradientDPSVI(model, guide, optimiser, C_aligned, dp_scale, N=N_train)
    if args.variant == "aligned" or args.variant == "all":
        print("aligned")
        aligned_dpsvi_params_for_epochs, aligned_dpsvi_grads_per_epochs = infer(
                aligned_dpsvi,
                lambda state, batch: aligned_dpsvi._compute_per_example_gradients(state, *batch)[2],
                (X_train, y_train),
                int(q * N_train),
                num_epochs,
                seed
        )
        filename = filenamer("aligned", "trace", C_aligned) + ".p"
        pd.to_pickle(aligned_dpsvi_params_for_epochs, filename)

    # pre-conditioned
    if args.variant == "precon" or args.variant == "all":
        print("precon")
        precon_dpsvi_params_for_epochs, precon_dpsvi_grads_per_epochs = infer(
                precon_dpsvi,
                preconditioned_grad_fn,
                (X_train, y_train),
                int(q * N_train),
                num_epochs,
                seed
        )
        filename = filenamer("precon", "trace", C_precon) + ".p"
        pd.to_pickle(precon_dpsvi_params_for_epochs, filename)


    # vanilla dpvi
    vanilla_dpsvi = VanillaDPSVI(model, guide, optimiser, C_vanilla, dp_scale, N=N_train)
    if args.variant == "vanilla" or args.variant == "all":
        print("vanilla")
        vanilla_dpsvi_params_for_epochs, vanilla_dpsvi_grads_per_epochs = infer(
                vanilla_dpsvi,
                lambda state, batch: vanilla_dpsvi._compute_per_example_gradients(state, *batch)[2],
                (X_train, y_train),
                int(q * N_train),
                num_epochs,
                seed
        )
        filename = filenamer("vanilla", "trace", C_vanilla) + ".p"
        pd.to_pickle(vanilla_dpsvi_params_for_epochs, filename)

    # # non-dp svi
    # if args.variant == "nondp":
    #     from numpyro.infer.svi import SVI
    #     from numpyro.infer import Trace_ELBO
    #     optimiser = Adam(1e-3)
    #     svi = SVI(model, guide, optimiser, Trace_ELBO(), N=N_train)
    #     nondp_svi = svi.run(jax.random.PRNGKey(123), int(1e5), X_train, y_train)
    #     nondp_svi_params = nondp_svi.params
    #     filename = f"{args.output_path}/adult_nondp_params_auto_scale{args.init_auto_scale}_T{nondp_svi.losses.shape[0]}_seed123_minmax{args.minmax_normalize}.p"
    #     pd.to_pickle(nondp_svi_params, filename)


if __name__=="__main__":
    main()
