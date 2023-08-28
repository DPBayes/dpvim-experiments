"""
Contains various helper/utility functions.
"""

import re
import numpy as np
import pandas as pd
import argparse
import scipy.stats as stats

import os
import pickle
from numpyro.infer.autoguide import AutoDiagonalNormal, biject_to
from numpyro.distributions.transforms import IdentityTransform

from collections import namedtuple
traces = namedtuple('traces', ['loc_trace', 'scale_trace'])

def filenamer(prefix, postfix, args=None, **kwargs):
    def filenamer_explicit(prefix, postfix, epsilon=None, clipping_threshold=None, k=None, seed=None, num_epochs=None, **unused_kwargs):
        output_name=f"{prefix}_{epsilon}_C{clipping_threshold}_k{k}_seed{seed}_epochs{num_epochs}_{postfix}"
        return output_name

    if isinstance(args, argparse.Namespace):
        new_kwargs = args.__dict__.copy()

        new_kwargs.update(kwargs)
        if 'prefix' in new_kwargs:
            del new_kwargs['prefix']
        kwargs = new_kwargs

    return filenamer_explicit(prefix, postfix, **kwargs)

def traces_to_dict(trace_tuple: traces):
    return {
        'auto_loc': trace_tuple.loc_trace,
        'auto_scale': trace_tuple.scale_trace
    }


def init_proportional_abs_error(dp_trace, nondp_trace, average=True):
    """ MPAE! """
    baseline = nondp_trace[-1]
    scale = np.abs(dp_trace[0]-baseline)
    errors = np.abs(dp_trace-baseline)
    scaled_errors = errors / scale
    if average:
        return scaled_errors.mean(1) # average over parameters
    else:
        return scaled_errors

### Figure styles
default_aspect = 12. / 9.
neurips_fig_style = {'figure.figsize':(5.5, 5.5 / default_aspect), 'font.size':10}
