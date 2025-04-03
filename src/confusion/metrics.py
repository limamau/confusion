from abc import abstractmethod

import jax.numpy as jnp
from jaxtyping import Array


class AbstractMetric:
    acronym: str

    @staticmethod
    @abstractmethod
    def __call__(gen_samples: Array, obs_samples: Array) -> Array:
        raise NotImplementedError


class MeanOfAbsMeanVarsDifference(AbstractMetric):
    """
    Mean of the absolute difference between the means of the variables.
    Variables are assumed to be spread across the first axis.
    """

    acronym: str = "MAMVD"

    @staticmethod
    def __call__(gen_samples: Array, obs_samples: Array) -> Array:
        gen_vars_means = jnp.mean(gen_samples, axis=0)
        obs_vars_means = jnp.mean(obs_samples, axis=0)
        return jnp.mean(jnp.abs(gen_vars_means - obs_vars_means))


class MeanOfAbsStdVarsDifference(AbstractMetric):
    """
    Mean of the absolute difference between the standard deviations of the variables.
    Variables are assumed to be spread across the first axis.
    """

    acronym: str = "MASVD"

    @staticmethod
    def __call__(gen_samples: Array, obs_samples: Array) -> Array:
        gen_vars_stds = jnp.std(gen_samples, axis=0)
        obs_vars_stds = jnp.std(obs_samples, axis=0)
        return jnp.mean(jnp.abs(gen_vars_stds - obs_vars_stds))
