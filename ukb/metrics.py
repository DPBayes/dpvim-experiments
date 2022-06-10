import numpy as np
import pandas as pd
from scipy import stats

def jaccard_index(orig_params, syn_params, alpha=0.05):
    n_dim = len(orig_params[0])
    dists = []
    for j in range(n_dim):
        orig_mean, orig_std = orig_params[0][j], orig_params[1][j]
        syn_mean, syn_std = syn_params[0][j], syn_params[1][j]

        orig_ci = stats.norm.ppf([alpha/2, 1.-alpha/2], loc=orig_mean, scale=orig_std)
        syn_ci = stats.norm.ppf([alpha/2, 1.-alpha/2], loc=syn_mean, scale=syn_std)

        # compute overlap
        overlap = max(0, min(orig_ci[1], syn_ci[1]) - max(orig_ci[0], syn_ci[0]))
        union = orig_ci[1] - orig_ci[0] + syn_ci[1] - syn_ci[0] - overlap

        dists.append(overlap/union)
    return pd.Series(dists, index=orig_params[0].index)


def f1ish(orig_params, syn_params, alpha=0.05):
    n_dim = len(orig_params[0])
    precisions = []
    recalls = []
    f1s = []
    for j in range(n_dim):
        orig_mean, orig_std = orig_params[0][j], orig_params[1][j]
        syn_mean, syn_std = syn_params[0][j], syn_params[1][j]

        orig_ci = stats.norm.ppf([alpha/2, 1.-alpha/2], loc=orig_mean, scale=orig_std)
        syn_ci = stats.norm.ppf([alpha/2, 1.-alpha/2], loc=syn_mean, scale=syn_std)

        # compute overlap
        overlap = max(0, min(orig_ci[1], syn_ci[1]) - max(orig_ci[0], syn_ci[0]))
        len_orig = orig_ci[1] - orig_ci[0]
        len_syn = syn_ci[1] - syn_ci[0]

        precision = overlap / len_syn
        recall = overlap / len_orig
        f1 = 2 * precision * recall / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    precisions = pd.Series(precisions, index=orig_params[0].index)
    recalls = pd.Series(recalls, index=orig_params[0].index)
    f1s = pd.Series(f1s, index=orig_params[0].index)
    return precisions, recalls, f1s


from numpyro.distributions import Normal
from numpyro.distributions.kl import kl_divergence
def sym_kl_div(orig_params, syn_params):
    n_dim = len(orig_params[0])
    kls = []
    for j in range(n_dim):
        orig_mean, orig_std = orig_params[0][j], orig_params[1][j]
        syn_mean, syn_std = syn_params[0][j], syn_params[1][j]

        orig_dist = Normal(orig_mean, orig_std)
        syn_dist = Normal(syn_mean, syn_std)

        kls.append(
            .5 * (kl_divergence(orig_dist, syn_dist) + kl_divergence(syn_dist, orig_dist))
        )


    kls = pd.Series(kls, index=orig_params[0].index)
    return kls

def init_proportional_abs_error(dp_trace, nondp_trace, average=True):
    baseline = nondp_trace[-1]
    scale = np.abs(dp_trace[0]-baseline)
    errors = np.abs(dp_trace-baseline)
    scaled_errors = errors / scale
    if average:
        return scaled_errors.mean(1) # average over parameters
    else:
        return scaled_errors
