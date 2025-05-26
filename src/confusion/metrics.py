import functools as ft
from abc import abstractmethod

import jax
import jax.numpy as jnp
from jaxtyping import Array


# helper functions #
def get_pdfs(
    gen_samples: Array, obs_samples: Array, bins: int = 10
) -> tuple[Array, Array]:
    min_range = jnp.minimum(jnp.min(gen_samples), jnp.min(obs_samples))
    max_range = jnp.maximum(jnp.max(gen_samples), jnp.max(obs_samples))
    len = jnp.shape(gen_samples)[0]
    gen_pdf, gen_edges = jnp.histogram(
        gen_samples,
        bins=bins,
        range=(min_range, max_range),
    )
    gen_pdf = gen_pdf / len
    obs_pdf, obs_edges = jnp.histogram(
        obs_samples,
        bins=bins,
        range=(min_range, max_range),
    )
    obs_pdf = obs_pdf / len
    return gen_pdf, obs_pdf


# metrics #
class AbstractMetric:
    acronym: str

    @abstractmethod
    def __call__(self, gen_samples: Array, obs_samples: Array) -> Array:
        raise NotImplementedError


class MeanOfAbsMeanVarsDifference(AbstractMetric):
    """
    Mean of the absolute difference between the means of the variables.
    Variables are assumed to be spread across the first axis.
    """

    acronym: str = "MAMVD"

    def __call__(self, gen_samples: Array, obs_samples: Array) -> Array:
        gen_vars_means = jnp.mean(gen_samples, axis=0)
        obs_vars_means = jnp.mean(obs_samples, axis=0)
        return jnp.mean(jnp.abs(gen_vars_means - obs_vars_means))


class MeanOfAbsStdVarsDifference(AbstractMetric):
    """
    Mean of the absolute difference between the standard deviations of the variables.
    Variables are assumed to be spread across the first axis.
    """

    acronym: str = "MASVD"

    def __call__(self, gen_samples: Array, obs_samples: Array) -> Array:
        gen_vars_stds = jnp.std(gen_samples, axis=0)
        obs_vars_stds = jnp.std(obs_samples, axis=0)
        return jnp.mean(jnp.abs(gen_vars_stds - obs_vars_stds))


class MaxMeanOrStdofAbsVarsDifference(AbstractMetric):
    """
    Maximum of the absolute difference between the means or standard deviation of the variables.
    Variables are assumed to be spread across the first axis.
    """

    acronym: str = "MMSAVD"

    def __call__(self, gen_samples: Array, obs_samples: Array) -> Array:
        gen_vars_means = jnp.mean(gen_samples, axis=0)
        obs_vars_means = jnp.mean(obs_samples, axis=0)
        max_mean = jnp.max(jnp.abs(gen_vars_means - obs_vars_means))
        gen_vars_std = jnp.std(gen_samples, axis=0)
        obs_vars_std = jnp.std(obs_samples, axis=0)
        max_std = jnp.max(jnp.abs(gen_vars_std - obs_vars_std))
        return jnp.max(jnp.array([max_mean, max_std]))


class WassersteinDistance(AbstractMetric):
    """Wasserstein Distance on L2."""

    acronym: str = "WD"

    def __init__(self, bins: int = 10):
        self.get_pdfs = ft.partial(get_pdfs, bins=bins)

    def __call__(self, gen_samples: Array, obs_samples: Array) -> Array:
        gen_pdfs, obs_pdfs = jax.vmap(self.get_pdfs)(gen_samples, obs_samples)
        wd = jnp.sqrt(jnp.mean(jnp.square(gen_pdfs - obs_pdfs)))
        return wd
