from numpyro import sample, plate
import numpyro.distributions as dists
from numpyro.infer.autoguide import AutoDiagonalNormal
import jax.numpy as jnp
import numpy as np
import pandas as pd
from twinify.mixture_model import MixtureModel
from twinify.model_loading import TModelFunction, TGuideFunction
import argparse
from typing import Iterable, Tuple
from collections import OrderedDict
from functools import partial

#ordinal_features = ["dAge", "iFertil", "iEnglish", "iImmigr", "iYearsch"]
categorical_features = ["dAncstry1", "dAncstry2", "iMarital"] + ["dAge", "iFertil", "iEnglish", "iImmigr", "iYearsch"]
binary_features = ["iSex", "iVietnam", "iKorean", "iMobillim"]
target = ["dPoverty"]

#all_features = ordinal_features + categorical_features + binary_features
all_features = categorical_features + binary_features

reference_groups = OrderedDict(
        dAncstry1  = 1,
        dAncstry2  = 1,
        iMarital   = 0,
        dAge       = 2,
        iFertil    = 0,
        iEnglish   = 0,
        iImmigr    = 0,
        iYearsch   = 1

    )

def preprocess(ori_df):
    df = pd.DataFrame()
    for feature in all_features:
        df[feature] = ori_df[feature].astype('category').cat.codes
    df["dPoverty"] = (ori_df.dPoverty.astype(int) < 2).map({False: 0, True: 1})
    y = df.pop('dPoverty')
    X = df[all_features]
    num_data = len(X)
    return (X, y), num_data

def postprocess(posterior_samples, ori_df):
    syn_X = posterior_samples['X']
    syn_y = posterior_samples['y']
    syn_data = jnp.hstack([syn_X, syn_y[:, np.newaxis]])
    columns = all_features.copy()
    columns.remove('dPoverty')
    columns.append('dPoverty')
    syn_df = pd.DataFrame(syn_data, columns = columns)

    encoded_syn_df = syn_df.copy()
    for feature in all_features:
        if feature != 'dPoverty':
            category_map = ori_df[feature].astype('category').cat.categories
            encoded_syn_df[feature] = pd.Categorical.from_codes(encoded_syn_df[feature], category_map)
    encoded_syn_df['dPoverty'] = encoded_syn_df['dPoverty'].astype(bool)

    return syn_df, encoded_syn_df

def onehot_encode(x, num_values):
    lut = jnp.eye(num_values)
    return lut[x]

def adjusted_onehot_encode(x, num_values, reference_idx):
    encoded = onehot_encode(x, num_values)
    return jnp.hstack((encoded[:, :reference_idx], encoded[:, (reference_idx+1):]))

def model_factory(twinify_args: argparse.Namespace, unparsed_args: Iterable[str], ori_df: pd.DataFrame) -> Tuple[TModelFunction, TGuideFunction]:
    model_args_parser = argparse.ArgumentParser("Model 1")
    model_args_parser.add_argument('--init_scale', type=float, default=0.1, help='Initial value for scales in variational AutoDiagonalNormal.')
    args = model_args_parser.parse_args(unparsed_args, twinify_args)

    k = args.k

    # collect number of categories and index of reference group/category for each feature
    d_sizes = OrderedDict()
    reference_indices = OrderedDict()
    for feature, reference_category in reference_groups.items():
        idx = list(ori_df[feature].astype('category').cat.categories).index(reference_category)
        reference_indices[feature] = idx
        d_sizes[feature] = len(ori_df[feature].astype('category').cat.categories)

    #for feature in ordinal_features:
    #    d_sizes[feature] = len(ori_df[feature].astype('category').cat.categories)

    def model(X=None, y=None, num_obs_total=None):
        d = len(all_features)
        batch_size = 1

        if y is not None:
            assert X is not None
            batch_size = X.shape[0]
            assert y.shape[0] == batch_size
            assert d == X.shape[1]
        else:
            assert X is None

        if num_obs_total is None:
            num_obs_total = batch_size

        ############## PRIORs ##################

        # Prior for mixture model of regressors
        prior_samples = []
        probs_for_mm = []
        for feature in all_features:
            if feature not in binary_features:
                prior_samples.append(sample(f'{feature}_probs', dists.Dirichlet(jnp.ones((k, d_sizes[feature])))))
                probs_for_mm.append(dists.Categorical(probs=prior_samples[-1]))
            else:
                prior_samples.append(sample(f'{feature}_probs', dists.Beta(jnp.ones((k,)), jnp.ones((k,)))))
                probs_for_mm.append(dists.Bernoulli(probs=prior_samples[-1]))

        pis = sample('pis_probs', dists.Dirichlet(jnp.ones(k)))

        mixture_dist = MixtureModel(probs_for_mm, pis)

        # Set up Poisson regression: this sets the prior for the regression weights
        # and also prepares functions that convert each categorical into (adjusted) one-hot vectors
        coefs = []
        feature_encoders = []
        for feature in all_features:
            if feature in categorical_features:
                adjusted_d_size = d_sizes[feature]- 1
                feature_coefs = sample(f"weight_{feature}", dists.Normal(jnp.zeros(adjusted_d_size), jnp.ones(adjusted_d_size)))
                coefs.append(feature_coefs)
                feature_encoders.append(partial(adjusted_onehot_encode, num_values=d_sizes[feature], reference_idx=reference_indices[feature]))

            else:
                feature_coefs = sample(f"weight_{feature}", dists.Normal(0.0, 1.0), sample_shape=(1,))
                feature_encoders.append(lambda x: x[:, jnp.newaxis])
                coefs.append(feature_coefs)

        weight = jnp.concatenate(coefs)

        intercept = sample("intercept", dists.Normal(0., 1.))

        ############# LIKELIHOOD ####################

        with plate('batch', num_obs_total, batch_size):
            X = sample('X', mixture_dist, obs=X)

            encoded_X = jnp.hstack([
                encode(X[:, i]) for i, encode in enumerate(feature_encoders)
            ])

            logit = encoded_X @ weight + intercept
            y = sample('y', dists.Poisson(rate=jnp.exp(logit)), obs=y)

    guide = AutoDiagonalNormal(model, init_scale=args.init_scale)

    return model, guide
