import pandas as pd
import numpy as np
import argparse
from typing import List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax.numpy as jnp
from utils import neurips_fig_style
plt.rcParams.update(neurips_fig_style)

parser = argparse.ArgumentParser(fromfile_prefix_chars="%")
parser.add_argument("variant", choices=('ng', 'vanilla', 'precon'))
parser.add_argument("--epsilon", type=float, default=None, help="Privacy level")
parser.add_argument("--dp_scale", type=float, default=None, help="DP-SGD noise std")
parser.add_argument("--num_epochs", "-e", default=4000, type=int, help="Number of training epochs.")
parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="Subsampling ratio for DP-SGD.")
parser.add_argument("--normalize", type=int, default=1)
parser.add_argument("--output_path", type=str, default="../results/", help="Path to folder in which to store parameter traces; defaults to '../results'")
parser.add_argument("--init_auto_scale", type=float, default=0.1, help="Initial std for the VI posterior")
parser.add_argument("--iteration", default=0, type=int)
args, unknown_args = parser.parse_known_args()


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

N_train = len(X_train)

C_ng = 0.1 # chosen by hand
C_vanilla = 3.0 # chosen by hand
C_aligned = 3.0 # chosen by hand
C_aligned_ng = 0.1 # chosen by hand
C_precon = 4.0 # chosen by hand

Cs = {
    'ng': C_ng,
    'vanilla': C_vanilla,
    'aligned': C_aligned,
    'aligned_ng': C_aligned_ng,
    'precon': C_precon
}

if args.epsilon is None and args.dp_scale is not None:
    dp_name_part = f"dp_scale{args.dp_scale}"
elif args.epsilon is not None:
    dp_name_part = f"eps{args.epsilon}"


seeds = range(123, 123+10)

q = args.sampling_ratio

## filename template
def filenamer(variant, suffix, C):
    return f"{args.output_path}/adult_{variant}_ne{args.num_epochs}_C{C}_q{q}_{dp_name_part}_auto_scale{args.init_auto_scale}_seed{seed}_optimadam_{suffix}"

##

from dpsvi import AlignedGradientDPSVI, AlignedNaturalGradientDPSVI, NaturalGradientDPSVI, PreconditionedGradientDPSVI, VanillaDPSVI

import jax
from numpyro.infer import Trace_ELBO, SVI
from numpyro.handlers import seed, substitute
import sys

sys.path.append("../ukb/")
from smart_auto_guide import SmartAutoGuide
from d3p.minibatch import subsample_batchify_data
import d3p.random

batch_size = int(q * len(X_train))
init_batchifier, get_batch = subsample_batchify_data((X_train, y_train), batch_size)
_, batchifier_state = init_batchifier(d3p.random.PRNGKey(17))
init_batch = get_batch(0, batchifier_state)

import numpyro.optim
optimiser = numpyro.optim.Adam(1e-3)

plot_df = pd.DataFrame()
site_cats = pd.CategoricalDtype(['auto_loc', 'auto_scale'])

# for seed in tqdm(seeds):

for seed in tqdm(seeds):
    # we interpret args.iteration 0 to look at parameters after initialisation with no optimsiation step done.
    # unfortunately the traces do not contain that, so we do some trickery here
    if args.iteration != 0:
        traces = pd.read_pickle(filenamer(args.variant, "trace", Cs[args.variant]) + ".p") # type: List[Dict[str, np.ndarray]]
        params = traces[args.iteration - 1 if args.iteration > 0 else args.iteration]
        guide = substitute(AutoDiagonalNormal(model), params)
    else:
        guide = AutoDiagonalNormal(model, init_scale=args.init_auto_scale)

    if args.variant == "vanilla":
        vanilla_dpsvi = VanillaDPSVI(model, guide, optimiser, Cs[args.variant], dp_scale=None, N=N_train)
        def px_grad_fn(state, batch):
            return vanilla_dpsvi._compute_per_example_gradients(state, *batch)[2]

        dpsvi_state = vanilla_dpsvi.init(d3p.random.PRNGKey(seed), *init_batch)
    elif args.variant == "precon":
        precon_dpsvi = PreconditionedGradientDPSVI(model, guide, optimiser, Cs[args.variant], dp_scale=None, N=N_train)
        def px_grad_fn(state, batch):
            px_grads_raw = precon_dpsvi._compute_per_example_gradients(state, *batch)[2]

            ## compute natural gradients
            params = precon_dpsvi.optim.get_params(state.optim_state)
            scale_transform_d = jax.vmap(jax.grad(lambda x: precon_dpsvi.constrain_fn(x)['auto_scale']))(params)['auto_scale']

            ## scale auto_scale gradients (back) up to same magnitude as auto_loc (because: g_{auto_scale} \approx T'(s) * g_{auto_loc})
            px_grads_raw['auto_scale'] = (1./scale_transform_d) * px_grads_raw['auto_scale']
            return px_grads_raw

        dpsvi_state = precon_dpsvi.init(d3p.random.PRNGKey(seed), *init_batch)
    elif args.variant == "ng":
        naturalgrad_dpsvi = NaturalGradientDPSVI(model, guide, optimiser, Cs[args.variant], dp_scale=None, N=N_train)
        def px_grad_fn(state, batch):

            ## compute vanilla gradients
            px_grads_raw = naturalgrad_dpsvi._compute_per_example_gradients(state, *batch)[2]

            ## compute natural gradients
            params = naturalgrad_dpsvi.optim.get_params(state.optim_state)
            constrained_params = naturalgrad_dpsvi.constrain_fn(params)
            scales = constrained_params['auto_scale']

            scale_transform_d = jax.vmap(jax.grad(lambda x: naturalgrad_dpsvi.constrain_fn(x)['auto_scale']))(params)['auto_scale']

            ## scale to natural gradient (assuming autodiagonalguide)
            px_grads_raw['auto_loc'] = scales**2 * px_grads_raw['auto_loc']
            px_grads_raw['auto_scale'] = .5 * (scales / scale_transform_d)**2 * px_grads_raw['auto_scale']

            return px_grads_raw

        dpsvi_state = naturalgrad_dpsvi.init(d3p.random.PRNGKey(seed), *init_batch)

    for i in tqdm(range(10), leave=False):
        batch = get_batch(i, batchifier_state)
        grads = px_grad_fn(dpsvi_state, batch)
        batch_df = pd.DataFrame()
        for site_name, site_grads in grads.items():
            batch_df = pd.concat((batch_df, pd.DataFrame({
                'px_grad_norm': np.linalg.norm(site_grads, axis=-1),
                'site': site_name
            })))
        batch_df['site'] = batch_df['site'].astype(site_cats)
        plot_df = pd.concat((plot_df, batch_df))

id_fn = lambda x: x
transform_fn = np.log # id_fn
fig, ax = plt.subplots()
bins = np.histogram_bin_edges(transform_fn(plot_df['px_grad_norm']), bins=100)
for site in ['auto_loc', 'auto_scale']:
    ax.hist(transform_fn(plot_df[plot_df['site'] == site]['px_grad_norm']), bins=bins, alpha=.8, label=site)

ax.set_xlabel("$\log(||g||_2)$")
ax.legend(['$g_m$', '$g_s$'])
fig.suptitle(f'Distribution of gradient norms\n{args.variant}')
# ax = sns.histplot(data=plot_df, x='px_grad_norm', hue='site')
fig.savefig(f"grad_norm_dist_{args.variant}_init{args.init_auto_scale}.pdf", format="pdf")
plt.close()