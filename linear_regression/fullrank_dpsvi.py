import jax, numpyro, d3p

import jax.numpy as jnp
import numpy as np

from jax import grad, vmap, jacobian

import d3p.random
from d3p.svi import full_norm

from numpyro.infer.svi import SVI, SVIState
from numpyro.infer.elbo import Trace_ELBO
from numpyro.handlers import seed
from numpyro.infer.util import log_density

import d3p.random as strong_rng
from d3p.svi import clip_gradient

softplus_transform = numpyro.distributions.biject_to(numpyro.distributions.constraints.softplus_positive)
exp_transform = numpyro.distributions.biject_to(numpyro.distributions.constraints.positive)

from dpsvi import PXGradientSVI

class AlignedGradientFullRankDPSVI(PXGradientSVI):
    """
    Full-rank covariance variant of the aligned SVI
    """

    def __init__(
            self,
            model,
            guide,
            optim,
            clipping_threshold, # set to None to scale norm to largest per-example gradient norm in the batchpx_clipped_grads
            dp_scale,
            rng_suite=strong_rng,
            **static_kwargs
        ):

        self._clipping_threshold = clipping_threshold
        self._dp_scale = dp_scale
        self._rng_suite = rng_suite
        self.static_kwargs = static_kwargs

        super().__init__(model, guide, optim, rng_suite, **static_kwargs)

    def mvn_entropy(self, unconstrained_params):
        """
        not quite the Entropy, but missing some constant coefficients
        """
        A = self.constrain_fn(unconstrained_params)['auto_scale_tril']
        return jnp.linalg.slogdet(A)[1]

    def _clip_gradients(self, dp_svi_state, px_gradients):
        """ Clips each per-example gradient.

        This is the second step in a full update iteration.

        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param px_gradients: Jax tuple tree of per-example gradients as returned
            by `_compute_per_example_gradients`
        :returns: tuple consisting of the updated svi state, a list of
            transformed per-example gradients per site and the jax tree structure
            definition. The list is a flattened representation of the jax tree,
            the shape of per-example gradients per parameter is unaffected.
        """
        # px_gradients is a jax tree of jax jnp.arrays of shape
        #   [batch_size, (param_shape)] for each parameter. flatten it out!
        px_grads_list, px_grads_tree_def = jax.tree_flatten(
            px_gradients
        )
        assert(len(px_grads_list)==2) # current implementation only for diagonal normal q
        assert('auto_loc' in px_gradients.keys() and 'auto_scale_tril' in px_gradients.keys())

        # clip only the loc gradient
        if self._clipping_threshold is not None:
            clipping_threshold = self._clipping_threshold
            px_clipped_loc_grads = vmap(clip_gradient, in_axes=(0, None))((px_gradients['auto_loc'],), self._clipping_threshold)[0]
        else:
            px_norms = vmap(jnp.linalg.norm, in_axes=0)(px_gradients['auto_loc'])
            clipping_threshold = jnp.max(px_norms)
            px_clipped_loc_grads = px_gradients['auto_loc']

        return dp_svi_state, px_clipped_loc_grads, px_grads_tree_def, clipping_threshold

    def update(self, orig_svi_state, *args, **kwargs):
        """ Takes a single step of SVI (possibly on a batch / minibatch of data),
        using the optimizer.

        :param svi_state: Current state of SVI.
        :param args: Arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: Keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: Tuple of `(svi_state, loss)`, where `svi_state` is the updated
            DPSVI state and `loss` the value of the ELBO on the training batch for
            the given `svi_state`.
        """

        ## compute the per-example gradients
        svi_state, per_example_loss, per_example_grads = \
            self._compute_per_example_gradients(orig_svi_state, *args, **kwargs)

        ## gather the MCMC draw used in ELBO estimation for the scale aligning
        # first replay the randomness
        _, d3p_rng_key = self._split_rng_key(orig_svi_state)
        jax_rng_key = self._rng_suite.convert_to_jax_rng_key(d3p_rng_key[0])
        _, guide_seed = jax.random.split(jax_rng_key)

        # replay the guide with the replayed randomness
        unconstrained_params = self.optim.get_params(svi_state.optim_state)
        seeded_guide = seed(self.guide, guide_seed)
        auto_scale_tril_jacobian_forward_fn = jax.vjp(
            lambda params: log_density(
                seeded_guide, args, self.static_kwargs, self.constrain_fn(params)
            )[1]['_auto_latent']['value'], unconstrained_params)[1]


        # clip gradients
        svi_state, px_clipped_loc_grads, tree_def, clipping_threshold = \
            self._clip_gradients(
                svi_state, per_example_grads
            )

        # compute avg elbo
        loss = jnp.mean(per_example_loss)

        # perturb the gradients
        avg_clipped_loc_grad = jnp.mean(px_clipped_loc_grads, axis=0)
        l2_sensitivity = clipping_threshold * (1. / self.batch_size)

        ###
        svi_state, step_rng_key = self._split_rng_key(svi_state)

        perturbation_scale = self._dp_scale * l2_sensitivity
        perturbed_loc_grad = self.perturbation_function(
            self._rng_suite, step_rng_key[0], (avg_clipped_loc_grad,), perturbation_scale
        )[0]

        perturbed_scale_tril_grad = auto_scale_tril_jacobian_forward_fn(perturbed_loc_grad)[0]["auto_scale_tril"]

        # add entropy contribution, scaled by 1/N
        perturbed_scale_tril_grad -= grad(self.mvn_entropy)(unconstrained_params)["auto_scale_tril"] / self.total_number_samples

        # collect
        perturbed_grads_dict = {'auto_loc': perturbed_loc_grad, 'auto_scale_tril': perturbed_scale_tril_grad}

        # take step
        svi_state = self._apply_gradient(svi_state, perturbed_grads_dict)

        return svi_state, loss

class VanillaFullRankDPSVI(PXGradientSVI):
    """
    Vanilla DPVI from Jälkö et al. 2017
    """

    def __init__(
            self,
            model,
            guide,
            optim,
            clipping_threshold, # set to None to scale norm to largest per-example gradient norm in the batchpx_clipped_grads
            dp_scale,
            rng_suite=strong_rng,
            **static_kwargs
        ):

        self._clipping_threshold = clipping_threshold
        self._dp_scale = dp_scale
        self._rng_suite = rng_suite
        self.static_kwargs = static_kwargs

        super().__init__(model, guide, optim, rng_suite, **static_kwargs)

    def _clip_gradients(self, dp_svi_state, px_gradients):
        """ Clips each per-example gradient.

        This is the second step in a full update iteration.

        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param px_gradients: Jax tuple tree of per-example gradients as returned
            by `_compute_per_example_gradients`
        :returns: tuple consisting of the updated svi state, a list of
            transformed per-example gradients per site and the jax tree structure
            definition. The list is a flattened representation of the jax tree,
            the shape of per-example gradients per parameter is unaffected.
        """
        # px_gradients is a jax tree of jax jnp.arrays of shape
        #   [batch_size, (param_shape)] for each parameter. flatten it out!
        px_grads_list, px_grads_tree_def = jax.tree_flatten(
            px_gradients
        )
        assert(len(px_grads_list)==2) # current implementation only for diagonal normal q
        assert('auto_loc' in px_gradients.keys() and 'auto_scale_tril' in px_gradients.keys())

        if self._clipping_threshold is not None:
            px_clipped_grads = vmap(clip_gradient, in_axes=(0, None))(px_grads_list, self._clipping_threshold)
            clipping_threshold = self._clipping_threshold
        else:
            px_norms = vmap(full_norm, in_axes=0)(px_grads_list)
            clipping_threshold = jnp.max(px_norms)
            px_clipped_grads = px_grads_list

        return dp_svi_state, px_clipped_grads, px_grads_tree_def, clipping_threshold

    def _perturb_and_reassemble_gradients(
            self, dp_svi_state, avg_clipped_grad, l2_sensitivity
        ):
        """ Perturbs the gradients using Gaussian noise and reassembles the gradient tree.

        This is the fourth step of a full update iteration.

        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param avg_clipped_grad: average clipped gradient
        :param l2_sensitivity: the l2 sensitivity for DP-SGD
        """

        dp_svi_state, step_rng_key = self._split_rng_key(dp_svi_state)

        perturbation_scale = self._dp_scale * l2_sensitivity
        perturbed_grads = self.perturbation_function(
            self._rng_suite, step_rng_key[0], avg_clipped_grad, perturbation_scale
        )

        perturbed_grads_dict = {
            'auto_loc': perturbed_grads[0],
            'auto_scale_tril': perturbed_grads[1]
        }

        return dp_svi_state, perturbed_grads_dict

    def update(self, orig_svi_state, *args, **kwargs):
        """ Takes a single step of SVI (possibly on a batch / minibatch of data),
        using the optimizer.

        :param svi_state: Current state of SVI.
        :param args: Arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: Keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: Tuple of `(svi_state, loss)`, where `svi_state` is the updated
            DPSVI state and `loss` the value of the ELBO on the training batch for
            the given `svi_state`.
        """

        ## compute the per-example gradients
        svi_state, per_example_loss, per_example_grads = \
            self._compute_per_example_gradients(orig_svi_state, *args, **kwargs)

        # clip gradients
        svi_state, px_clipped_grads, tree_def, clipping_threshold = \
            self._clip_gradients(
                svi_state, per_example_grads
            )

        # compute avg elbo
        loss = jnp.mean(per_example_loss)

        # perturb the gradients
        avg_clipped_grad = tuple(jnp.mean(site, axis=0) for site in px_clipped_grads)
        l2_sensitivity = clipping_threshold * (1. / self.batch_size)
        svi_state, gradient = self._perturb_and_reassemble_gradients(
            svi_state, avg_clipped_grad, l2_sensitivity
        )

        # take step
        svi_state = self._apply_gradient(svi_state, gradient)

        return svi_state, loss
