import functools as ft
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


class StandardWeighting(AbstractWeighting):
    def __init__(self, sde: AbstractSDE):
        self.sde = sde

    def __call__(self, t: Array) -> Array:
        return self.sde.sigma(t) ** 2


class EDMWeighting(AbstractWeighting):
    def __init__(self, sde: AbstractSDE, sigma_data: float):
        self.sde = sde
        self.sigma_data = sigma_data

    def __call__(self, t: Array) -> Array:
        return (self.sde.sigma(t) ** 2 + self.sigma_data**2) / (
            self.sde.sigma(t) * self.sigma_data
        ) ** 2


# losses #
class ScoreMatchingLoss:
    def __init__(
        self,
        weighting: AbstractWeighting,
        std_clip: float = 1e-5,
    ):
        self.weighting = weighting
        self.std_clip = std_clip

    def single_loss_fn(
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
        noise = jr.normal(key, x0.shape)
        x = mean + std * noise
        pred = model.score(x, t, c, key=dropout_key)
        return self.weighting(t) * jnp.mean((pred + noise / std) ** 2)

    def __call__(
        self,
        model: AbstractDiffusionModel,
        x0: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        loss_fn = ft.partial(self.single_loss_fn, model)
        loss_fn = jax.vmap(loss_fn)
        return jnp.mean(loss_fn(x0, t, c, key))
