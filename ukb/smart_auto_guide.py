from typing import Any, Callable, Iterable, Optional, Type
from numpyro.infer.autoguide import AutoGuide, AutoGuide
from numpyro.handlers import condition, trace, seed, block
import jax.random

__all__ = ['SmartAutoGuide']

class SmartAutoGuide(AutoGuide):
    """
    Wraps any given AutoGuide class and allows it to be used in sampling without
    first running inference, given only a list of known observation sites in the model
    and previously learned parameters.

    The main intent is to facilitate sampling from the guide in a separate
    script than the inference took place, i.e., when the particular guide instance
    was not used for inference, which is not possible with regular AutoGuide instances.
    """

    def __init__(self,
            model: Callable,
            observation_sites: Optional[Iterable[str]],
            base_guide_class: Type[AutoGuide],
            *base_guide_args: Any,
            **base_guide_kwargs: Any) -> None:
        """
        Creates a SmartAutoGuide for the given `base_guide_class` AutoGuide. If `observation_sites`
        are not `None` and covers all sampling sites which `model` was conditioned on, i.e., those
        which should not be affected by `base_guide_class`, the created SmartAutoGuide can be used
        for sampling without being used in inference first.

        If `observation_sites` are `None`, using the created SmartAutoGuide in inference
        will behave exactly as using an instance of `base_guide_class` but will additionally
        extract the the sampling sites which `model` is conditioned on during inference and
        make them avaiable via the `observation_sites` property. These can then be used
        to initialise SmartAutoGuide instances for sampling without inference later on.

        :param model: The model functions.
        :param observation_sites: Collection of parameter site names that are observations in the model.
        :param base_guide_class: The AutoGuide subclass to wrap around (NOT a class instance!).
        :param base_guide_args: Positional arguments to pass into base_guide_class's initializer.
        :param base_guide_kwargs: Keyword arguments to pass into bas_guide_class's initializer.
        """
        self._model = model
        self._base_guide_factory = lambda model: base_guide_class(model, *base_guide_args, **base_guide_kwargs)
        self._obs = frozenset(observation_sites) if observation_sites is not None else None
        self._guide = None

    @staticmethod
    def wrap_for_inference(base_guide_class: Type[AutoGuide]) -> Callable[[Any], "SmartAutoGuide"]:
        def wrapped_for_inference(model: Callable, *args, **kwargs):
            return SmartAutoGuide(model, None, base_guide_class, *args, **kwargs)
        return wrapped_for_inference

    @staticmethod
    def wrap_for_sampling(base_guide_class: Type[AutoGuide], observation_sites: Iterable[str]) -> Callable[[Any], "SmartAutoGuide"]:
        def wrapped_for_sampling(model: Callable, *args, **kwargs):
            return SmartAutoGuide(model, observation_sites, base_guide_class, *args, **kwargs)
        return wrapped_for_sampling

    @staticmethod
    def wrap_for_sampling_and_initialize(
            base_guide_class: Type[AutoGuide],
            observation_sites: Iterable[str],
            *model_args: Any, **model_kwargs) -> Callable[[Any], "SmartAutoGuide"]:
        def wrapped_for_sampling_with_init(model: Callable, *base_guide_args, **base_guide_kwargs):
            guide = SmartAutoGuide(
                model, observation_sites, base_guide_class, *base_guide_args, **base_guide_kwargs
            )
            guide.initialize(*model_args, **model_kwargs)
            return guide
        return wrapped_for_sampling_with_init

    def initialize(self, *args: Any, **kwargs: Any) -> Any:
        if self._guide is not None: return

        if self._obs is not None:
            # We are given a set of observation sites that should be ignored by the guide
            # Here's the general strategy: NumPyro AutoGuide requires that all
            # parameters sites corresponding to observations need to be conditioned
            # on, but this is not case yet. To do so, we sample some data from the
            # prior predictive distribution (using the model function) and condition
            # the model on those for the guide initialisation.

            # We will probably be in the middle of a sampling stack so we block
            # all handlers sitting above the guide while we initialize it
            with block():

                # get plausible fake values for observed sites from prior
                with trace() as tr:
                    seed(self._model, jax.random.PRNGKey(0))(*args, **kwargs)
                    fake_obs = {site: val['value'] for site, val in tr.items() if site in self._obs}

                # feed model conditioned on fake observations to guide
                guide = self._base_guide_factory(condition(self._model, fake_obs))
                # trigger guide initialisation with fake observatons
                seed(guide, jax.random.PRNGKey(0))(*args, **kwargs)

            self._guide = guide
        else:
            # We are not given a set of observation sites, so we assume this call is
            # part of inference. We don't need to do anything to the guide.
            # However, we trace through the model to collect all observation sites
            # that it is conditioned on.

            with block():
                with trace() as tr:
                    seed(self._model, jax.random.PRNGKey(0))(*args, **kwargs)
                self._obs = [name for name, site in tr.items() if site.get('is_observed', False)]

            guide = self._base_guide_factory(self._model) # initialise guide with model as normal
            self._guide = guide

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._guide is None:
            self.initialize(*args, **kwargs)

        return self._guide.__call__(*args, **kwargs)

    def sample_posterior(self, rng_key, params, *args, **kwargs):
        return self.base_guide.sample_posterior(rng_key, params, *args, **kwargs)

    @property
    def observation_sites(self) -> Iterable[str]:
        return self._obs

    def median(self, params):
        """ Note: Requires initialization. """
        return self.base_guide.median(params)

    def quantiles(self, params, quantiles):
        """ Note: Requires initialization. """
        return self.base_guide.quantiles(params, quantiles)

    @property
    def base_guide(self) -> AutoGuide:
        if not self.is_initialized:
            raise RuntimeError("The guide must be initialized with from the model first! "\
                "You can call initialize(*model_args, **model_kwargs) to do so.")
        return self._guide

    def _unpack_latent(self, x):
        """ Note: Requires initialization and that the `base_guide_class` provided upon
        construction of the SmartAutoGuide is of type AutoContinuous, otherwise errors. """
        return self.base_guide._unpack_latent(x)

    @property
    def is_initialized(self) -> bool:
        return self._guide is not None
