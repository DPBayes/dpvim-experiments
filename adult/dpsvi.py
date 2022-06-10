import jax, numpyro, d3p

import jax.numpy as jnp
import numpy as np

from jax import grad, vmap
from jax.lax import fori_loop

from d3p.svi import DPSVI
import d3p.random

from typing import Any, NamedTuple, Sequence, Tuple

from numpyro.infer.svi import SVI, SVIState
from numpyro.infer.elbo import Trace_ELBO
from numpyro.handlers import seed, trace, substitute, block
from numpyro.infer.util import get_importance_trace, log_density

from d3p.util import example_count
import d3p.random as strong_rng
from d3p.svi import DPSVIState, full_norm, normalize_gradient, get_gradients_clipping_function, clip_gradient, get_observations_scale

from collections import namedtuple

PRNGState = Any
softplus_transform = numpyro.distributions.biject_to(numpyro.distributions.constraints.softplus_positive)
exp_transform = numpyro.distributions.biject_to(numpyro.distributions.constraints.positive)


class PXGradientSVI(SVI):
    """
    This class defines the basic functionality of per example gradient ELBO.
    """

    def __init__(
            self,
            model,
            guide,
            optim,
            rng_suite=strong_rng,
            **static_kwargs
        ):

        self._rng_suite = rng_suite
        self.static_kwargs = static_kwargs

        super().__init__(model, guide, optim, Trace_ELBO(), **static_kwargs)

    @staticmethod
    def _update_state_rng(dp_svi_state: DPSVIState, rng_key: PRNGState) -> DPSVIState:
        return DPSVIState(
            dp_svi_state.optim_state,
            rng_key,
            dp_svi_state.observation_scale
        )

    @staticmethod
    def _update_state_optim_state(dp_svi_state: DPSVIState, optim_state: Any) -> DPSVIState:
        return DPSVIState(
            optim_state,
            dp_svi_state.rng_key,
            dp_svi_state.observation_scale
        )

    def _split_rng_key(self, dp_svi_state: DPSVIState, count: int = 1) -> Tuple[DPSVIState, Sequence[PRNGState]]:
        rng_key = dp_svi_state.rng_key
        split_keys = self._rng_suite.split(rng_key, count+1)
        return DPSVI._update_state_rng(dp_svi_state, split_keys[0]), split_keys[1:]

    def init(self, rng_key, *args, **kwargs):
        svi_state = super().init(self._rng_suite.convert_to_jax_rng_key(rng_key), *args, **kwargs)

        # infer the total number of samples
        params = self.optim.get_params(svi_state.optim_state)

        model_kwargs = dict(kwargs)
        model_kwargs.update(self.static_kwargs)

        one_element_batch = [
            jnp.expand_dims(a[0], 0) for a in args
        ]

        observation_scale = get_observations_scale(
            self.model, one_element_batch, model_kwargs, params
        )
        self.total_number_samples = len(one_element_batch[0]) * observation_scale
        self.batch_size = jnp.shape(args[0])[0]

        return DPSVIState(svi_state.optim_state, rng_key, observation_scale=1.)

    def _compute_per_example_gradients(self, dp_svi_state, *args, **kwargs):
        """ Computes the raw per-example gradients of the model.

        This is the first step in a full update iteration.

        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param args: Arguments to the loss function.
        :param kwargs: All keyword arguments to model or guide.
        :returns: tuple consisting of the updated DPSVI state, an array of loss
            values per example, and a jax tuple tree of per-example gradients
            per parameter site (each site's gradients have shape (batch_size, *parameter_shape))
        """
        dp_svi_state, d3p_rng_key = self._split_rng_key(dp_svi_state)
        jax_rng_key = self._rng_suite.convert_to_jax_rng_key(d3p_rng_key[0])

        params = self.optim.get_params(dp_svi_state.optim_state)

        # we wrap the per-example loss (ELBO) to make it easier "digestable"
        # for jax.vmap(jax.value_and_grad()): slighly reordering parameters; fixing kwargs, model and guide
        def wrapped_px_loss(prms, rng_key, loss_args):
            # vmap removes leading dimensions, we re-add those in a wrapper for fun so
            # that fun can be oblivious of this
            new_args = (jnp.expand_dims(arg, 0) for arg in loss_args)
            return self.loss.loss(
                rng_key, self.constrain_fn(prms), self.model, self.guide,
                *new_args, **kwargs, **self.static_kwargs
            )

        px_value_and_grad = jax.vmap(jax.value_and_grad(wrapped_px_loss), in_axes=(None, None, 0))
        per_example_loss, per_example_grads = px_value_and_grad(params, jax_rng_key, args)

        # the scaling of likelihood contributions with total number of samples makes it difficult to
        # keep clipping comparable between data sets of different size. Thus we will scale our gradients
        # by 1/N, where N is the total number of samples. NOTE that we need to reflect this also to the
        # gradient of the entropy later!
        per_example_grads = {key: value / self.total_number_samples for key, value in per_example_grads.items()}

        return dp_svi_state, per_example_loss, per_example_grads

    def _apply_gradient(self, dp_svi_state, batch_gradient):
        """ Takes a (batch) gradient step in parameter space using the specified
            optimizer.

        This is the fifth and last step of a full update iteration.
        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param batch_gradient: Jax tree of batch gradients per parameter site,
            as returned by `_combine_and_transform_gradient`.
        :returns: tuple consisting of the updated svi state.
        """
        optim_state = dp_svi_state.optim_state
        new_optim_state = self.optim.update(batch_gradient, optim_state)

        dp_svi_state = self._update_state_optim_state(dp_svi_state, new_optim_state)
        return dp_svi_state

    def evaluate(self, svi_state: DPSVIState, *args, **kwargs):
        """ Evaluates the ELBO given the current parameter values / DPSVI state
        and (a minibatch of) data.

        :param svi_state: Current state of DPSVI.
        :param args: Arguments to the model / guide.
        :param kwargs: Keyword arguments to the model / guide.
        :return: ELBO loss given the current parameter values
            (held within `svi_state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given an svi_state
        jax_rng_key = self._rng_suite.convert_to_jax_rng_key(self._rng_suite.split(svi_state.rng_key, 1)[0])
        numpyro_svi_state = SVIState(svi_state.optim_state, None, jax_rng_key)
        return super().evaluate(numpyro_svi_state, *args, **kwargs)

    @staticmethod
    def perturbation_function(
            rng_suite, rng: PRNGState, values: Sequence[jnp.ndarray], perturbation_scale: float
        ) -> Sequence[jnp.ndarray]:  # noqa: E121, E125
        """ Perturbs given values using Gaussian noise.

        `values` can be a list of array-like objects. Each value is independently
        perturbed by adding noise sampled from a Gaussian distribution with a
        standard deviation of `perturbation_scale`.

        :param rng: Jax PRNGKey for perturbation randomness.
        :param values: Iterable of array-like where each value will be perturbed.
        :param perturbation_scale: The scale/standard deviation of the noise
            distribution.
        """
        def perturb_one(a: jnp.ndarray, site_rng: PRNGState) -> jnp.ndarray:
            """ perturbs a single gradient site """
            noise = rng_suite.normal(site_rng, a.shape) * perturbation_scale
            return a + noise

        per_site_rngs = rng_suite.split(rng, len(values))
        values = tuple(
            perturb_one(grad, site_rng)
            for grad, site_rng in zip(values, per_site_rngs)
        )
        return values

    def get_params(self, svi_state: DPSVIState):
        return self.constrain_fn(self.optim.get_params(svi_state.optim_state))

class AlignedGradientDPSVI(PXGradientSVI):
    """
    DPSVI pipeline where we have aligned the mu and loc gradients after perturbation. WORKS ONLY for diagonal normal variational posteriors.
    Note that ELBO can be factorized as follows
        ELBO(q) := L(q) = E_q[log p(X, Z)] + H(q)
    Now, using the reparametrization trick, we have Z := g(eta) = q_m + eta * T(q_s), where T denotes a function that
    translates unconstrained parameter q_s to the variational std.
    Now, if we use MC integration with a single sample to estimate the expectations, we have
        d/dq_m L(q) ~= d/dq_m log p(X, g(eta))
        d/dq_s L(q) ~= eta * T'(s) d/dq_m log p(X, g(eta)) = eta * T'(s) * d/dq_m L(q)
    Thus we only need to perturb the d/dq_m L(q) and reconstruct the d/dq_s L(q) based on that
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
        assert('auto_loc' in px_gradients.keys() and 'auto_scale' in px_gradients.keys())

        # clip only the loc gradient
        if self._clipping_threshold is not None:
            clipping_threshold = self._clipping_threshold
            px_clipped_loc_grads = vmap(clip_gradient, in_axes=(0, None))((px_gradients['auto_loc'],), self._clipping_threshold)[0]
        else:
            px_norms = vmap(jnp.linalg.norm, in_axes=0)(px_gradients['auto_loc'])
            clipping_threshold = jnp.max(px_norms)
            px_clipped_loc_grads = px_gradients['auto_loc']

        return dp_svi_state, px_clipped_loc_grads, px_grads_tree_def, clipping_threshold

    def _perturb_and_reassemble_gradients(
            self, dp_svi_state, avg_clipped_loc_grad, l2_sensitivity, guide_trace
        ):
        """ Perturbs the gradients using Gaussian noise and reassembles the gradient tree.

        This is the fourth step of a full update iteration.

        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param avg_clipped_loc_grad: average clipped location gradient
        :param l2_sensitivity: the l2 sensitivity for DP-SGD
        """

        dp_svi_state, step_rng_key = self._split_rng_key(dp_svi_state)

        perturbation_scale = self._dp_scale * l2_sensitivity
        perturbed_loc_grad = self.perturbation_function(
            self._rng_suite, step_rng_key[0], (avg_clipped_loc_grad,), perturbation_scale
        )[0]

        ## assemble scale grad
        unconstrained_params = self.optim.get_params(dp_svi_state.optim_state)
        scales = guide_trace['auto_scale']['value']
        draw = (guide_trace['_auto_latent']['value'] - guide_trace['auto_loc']['value']) / scales
        scale_transform_d = vmap(grad(lambda x: self.constrain_fn(x)['auto_scale']))(unconstrained_params)['auto_scale']

        perturbed_scale_grad = scale_transform_d * draw * perturbed_loc_grad

        # add entropy contribution, scaled by 1/N
        perturbed_scale_grad -=  (scale_transform_d / scales) / self.total_number_samples

        # collect
        perturbed_grads_dict = {'auto_loc': perturbed_loc_grad, 'auto_scale': perturbed_scale_grad}

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

        ## gather the MCMC draw used in ELBO estimation for the scale aligning
        # first replay the randomness
        _, d3p_rng_key = self._split_rng_key(orig_svi_state)
        jax_rng_key = self._rng_suite.convert_to_jax_rng_key(d3p_rng_key[0])
        _, guide_seed = jax.random.split(jax_rng_key)

        # replay the guide with the replayed randomness
        params = self.optim.get_params(svi_state.optim_state)
        seeded_guide = seed(self.guide, guide_seed)
        _, guide_trace = log_density(
            seeded_guide, args, self.static_kwargs, self.constrain_fn(params)
        )

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
        svi_state, gradient = self._perturb_and_reassemble_gradients(
            svi_state, avg_clipped_loc_grad, l2_sensitivity, guide_trace
        )

        # take step
        svi_state = self._apply_gradient(svi_state, gradient)

        return svi_state, loss

class AlignedNaturalGradientDPSVI(PXGradientSVI):
    """
    DPSVI pipeline where we use natural gradients AND have aligned the mu and loc gradients after perturbation. WORKS ONLY for diagonal normal variational posteriors.
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
        assert('auto_loc' in px_gradients.keys() and 'auto_scale' in px_gradients.keys())

        # clip only the loc gradient
        if self._clipping_threshold is not None:
            px_clipped_loc_grads = vmap(clip_gradient, in_axes=(0, None))((px_gradients['auto_loc'],), self._clipping_threshold)[0]
            clipping_threshold = self._clipping_threshold
        else:
            px_norms = vmap(jnp.linalg.norm, in_axes=0)(px_gradients['auto_loc'])
            clipping_threshold = jnp.max(px_norms)
            px_clipped_loc_grads = px_gradients['auto_loc']

        return dp_svi_state, px_clipped_loc_grads, px_grads_tree_def, clipping_threshold

    def _perturb_and_reassemble_gradients(
            self, dp_svi_state, avg_clipped_loc_grad, l2_sensitivity, guide_trace
        ):
        """ Perturbs the gradients using Gaussian noise and reassembles the gradient tree.

        This is the fourth step of a full update iteration.

        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param avg_clipped_loc_grad: average clipped location gradient
        :param l2_sensitivity: the l2 sensitivity for DP-SGD
        """

        dp_svi_state, step_rng_key = self._split_rng_key(dp_svi_state)

        perturbation_scale = self._dp_scale * l2_sensitivity
        perturbed_loc_natural_grad = self.perturbation_function(
            self._rng_suite, step_rng_key[0], (avg_clipped_loc_grad,), perturbation_scale
        )[0]

        ## assemble scale grad
        unconstrained_params = self.optim.get_params(dp_svi_state.optim_state)
        scales = guide_trace['auto_scale']['value']
        draw = (guide_trace['_auto_latent']['value'] - guide_trace['auto_loc']['value']) / scales
        scale_transform_d = vmap(grad(lambda x: self.constrain_fn(x)['auto_scale']))(unconstrained_params)['auto_scale']

        perturbed_loc_grad = 1/scales**2 * perturbed_loc_natural_grad
        perturbed_scale_grad = scale_transform_d * draw * perturbed_loc_grad

        # add entropy contribution, scaled by 1/N
        perturbed_scale_grad -=  (scale_transform_d / scales) / self.total_number_samples

        ## scale to natural gradient (assuming autodiagonalguide)
        scales = guide_trace['auto_scale']['value']
        perturbed_scale_natural_grad = .5 * (scales / scale_transform_d)**2 * perturbed_scale_grad

        perturbed_grads_dict = {'auto_loc': perturbed_loc_natural_grad, 'auto_scale': perturbed_scale_natural_grad}

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


        ## gather the MCMC draw used in ELBO estimation for the scale aligning
        # first replay the randomness
        _, d3p_rng_key = self._split_rng_key(orig_svi_state)
        jax_rng_key = self._rng_suite.convert_to_jax_rng_key(d3p_rng_key[0])
        _, guide_seed = jax.random.split(jax_rng_key)

        # replay the guide with the replayed randomness
        params = self.optim.get_params(svi_state.optim_state)
        seeded_guide = seed(self.guide, guide_seed)
        _, guide_trace = log_density(
            seeded_guide, args, self.static_kwargs, self.constrain_fn(params)
        )

        ## compute natural gradients
        constrained_params = self.constrain_fn(params)
        scales = constrained_params['auto_scale']

        scale_transform_d = vmap(grad(lambda x: self.constrain_fn(x)['auto_scale']))(params)['auto_scale']

        ## scale to natural gradient (assuming autodiagonalguide)
        per_example_grads['auto_loc'] = scales**2 * per_example_grads['auto_loc']
        per_example_grads['auto_scale'] = .5 * (scales / scale_transform_d)**2 * per_example_grads['auto_scale']

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
        svi_state, gradient = self._perturb_and_reassemble_gradients(
            svi_state, avg_clipped_loc_grad, l2_sensitivity, guide_trace
        )

        # take step
        svi_state = self._apply_gradient(svi_state, gradient)

        return svi_state, loss

class NaturalGradientDPSVI(PXGradientSVI):
    """
    DPSVI pipeline with natural gradients before clipping and perturbation.
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
        assert('auto_loc' in px_gradients.keys() and 'auto_scale' in px_gradients.keys())

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
            'auto_scale': perturbed_grads[1]
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

        ## compute natural gradients
        params = self.optim.get_params(svi_state.optim_state)
        constrained_params = self.constrain_fn(params)
        scales = constrained_params['auto_scale']

        scale_transform_d = vmap(grad(lambda x: self.constrain_fn(x)['auto_scale']))(params)['auto_scale']

        ## scale to natural gradient (assuming autodiagonalguide)
        per_example_grads['auto_loc'] = scales**2 * per_example_grads['auto_loc']
        per_example_grads['auto_scale'] = .5 * (scales / scale_transform_d)**2 * per_example_grads['auto_scale']

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


class PreconditionedGradientDPSVI(PXGradientSVI):
    """
    DPSVI pipeline that scales scale gradients before clipping and perturbation.
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
        assert('auto_loc' in px_gradients.keys() and 'auto_scale' in px_gradients.keys())

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
            'auto_scale': perturbed_grads[1]
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

        ## compute natural gradients
        params = self.optim.get_params(svi_state.optim_state)
        scale_transform_d = vmap(grad(lambda x: self.constrain_fn(x)['auto_scale']))(params)['auto_scale']

        ## scale auto_scale gradients (back) up to same magnitude as auto_loc (because: g_{auto_scale} \approx T'(s) * g_{auto_loc})
        per_example_grads['auto_scale'] = (1./scale_transform_d) * per_example_grads['auto_scale']

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

        ## scale auto_scale gradients back down to their "proper" magnitude
        gradient['auto_scale'] = scale_transform_d * gradient['auto_scale']

        # take step
        svi_state = self._apply_gradient(svi_state, gradient)

        return svi_state, loss


class VanillaDPSVI(PXGradientSVI):
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
        assert('auto_loc' in px_gradients.keys() and 'auto_scale' in px_gradients.keys())

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
            'auto_scale': perturbed_grads[1]
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
