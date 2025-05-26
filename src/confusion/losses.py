import functools as ft
from abc import abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .diffusion import AbstractDiffusionModel
from .sdes import AbstractSDE


# weighting used by loss #
class AbstractWeighting:
    def __call__(self, t: Array) -> Array:
        raise NotImplementedError


class NoWeighting(AbstractWeighting):
    def __init__(self):
        pass

    def __call__(self, t: Array) -> Array:
        return jnp.ones_like(t)


class SqrtWeighting(AbstractWeighting):
    def __init__(self, sde: AbstractSDE):
        self.sde = sde

    def __call__(self, t: Array) -> Array:
        return jnp.sqrt(self.sde.sigma(t))


class StandardWeighting(AbstractWeighting):
    def __init__(self, sde: AbstractSDE):
        self.sde = sde

    def __call__(self, t: Array) -> Array:
        return self.sde.sigma(t)


class InverseSqrtWeighting(AbstractWeighting):
    def __init__(self, sde: AbstractSDE, factor: float = 1e3):
        self.sde = sde
        self.factor = factor

    def __call__(self, t: Array) -> Array:
        return jnp.sqrt(1 / self.sde.sigma(t) / self.factor)


class InverseWeighting(AbstractWeighting):
    def __init__(self, sde: AbstractSDE, factor: float = 1e3):
        self.sde = sde
        self.factor = factor

    def __call__(self, t: Array) -> Array:
        return 1 / self.sde.sigma(t) / self.factor


class EDMWeighting(AbstractWeighting):
    def __init__(self, sde: AbstractSDE, sigma_data: float):
        self.sde = sde
        self.sigma_data = sigma_data

    def __call__(self, t: Array) -> Array:
        num = jnp.sqrt(self.sde.sigma(t) ** 2 + self.sigma_data**2)
        den = self.sde.sigma(t) * self.sigma_data
        return num / den


# losses #
class AbstractLoss:
    weighting: AbstractWeighting
    std_clip: float

    @abstractmethod
    def __call__(
        self,
        model: AbstractDiffusionModel,
        x0: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        raise NotImplementedError


class ScoreMatchingLoss(AbstractLoss):
    def __init__(
        self,
        weighting: AbstractWeighting,
        std_clip: float = 1e-5,
    ):
        self.weighting = weighting
        self.std_clip = std_clip

    def single_score_matching_fn(
        self,
        model: AbstractDiffusionModel,
        x0: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        noise_key, dropout_key = jr.split(key)
        mean, std = model.sde.perturbation(x0, t)
        # clip std to avoid division by zero
        std = jnp.maximum(std, self.std_clip)
        standard_noise = jr.normal(key, x0.shape)
        x = mean + std * standard_noise
        score = model.score(x, t, c, key=dropout_key)
        return jnp.mean(jnp.square(self.weighting(t) * (score + standard_noise / std)))

    def __call__(
        self,
        model: AbstractDiffusionModel,
        x0: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        loss_fn = ft.partial(self.single_score_matching_fn, model)
        loss_fn = jax.vmap(loss_fn)
        return jnp.mean(loss_fn(x0, t, c, key))
