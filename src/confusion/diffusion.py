from abc import abstractmethod
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .networks import AbstractNetwork
from .sdes import AbstractSDE
from .weighting import AbstractWeighting


class AbstractDiffusionModel(eqx.Module):
    sde: AbstractSDE

    def __init__(self, sde: AbstractSDE):
        self.sde = sde

    @abstractmethod
    def score(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def loss(
        self,
        x0: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        raise NotImplementedError


class StandardDiffusionModel(AbstractDiffusionModel):
    network: AbstractNetwork
    weighting: AbstractWeighting
    sde: AbstractSDE

    def __init__(
        self,
        network: AbstractNetwork,
        weighting: AbstractWeighting,
        sde: AbstractSDE,
    ):
        self.network = network
        self.weighting = weighting
        super().__init__(sde)

    def loss(
        self,
        x0: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        noise_key, dropout_key = jr.split(key)
        mean, std = self.sde.perturbation(x0, t)
        standard_noise = jr.normal(key, x0.shape)
        x = mean + std * standard_noise
        net = self.network(x, t, c, key=dropout_key)
        weight_noise_ratio = self.weighting(t) / jnp.square(std)
        noise_loss = jnp.mean(jnp.square((net + standard_noise)))
        return weight_noise_ratio * noise_loss

    def score(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        return self.network(x, t, c, key=key) / self.sde.sigma(t)


class DenoiserDiffusionModel(AbstractDiffusionModel):
    network: AbstractNetwork
    weighting: AbstractWeighting
    sde: AbstractSDE
    sigma_data: float

    def __init__(
        self,
        network: AbstractNetwork,
        weighting: AbstractWeighting,
        sde: AbstractSDE,
        sigma_data: float,
    ):
        self.network = network
        self.weighting = weighting
        super().__init__(sde)
        self.sigma_data = sigma_data

    def denoise(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        sigma = self.sde.sigma(t)
        c_skip = self.sigma_data**2 / (self.sigma_data**2 + sigma**2)
        c_out = self.sigma_data * sigma / jnp.sqrt(self.sigma_data**2 + sigma**2)
        c_in = 1 / jnp.sqrt(self.sigma_data**2 + sigma**2)
        c_noise = 0.25 * jnp.log(sigma)
        left_side = c_skip * x
        net_x = c_in * x
        net_sigma = c_noise * sigma
        right_side = c_out * self.network(net_x, net_sigma, c, key=key)
        return left_side + right_side

    def loss(
        self,
        x0: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        noise_key, dropout_key = jr.split(key)
        # limamau: take out the perturbation method as this is only used in the standard diffusion model
        # limamau: also go from s to mu in notation/method
        std = self.sde.sigma(t)
        standard_noise = jr.normal(key, x0.shape)
        x = x0 + std * standard_noise
        denoiser = self.denoise(x, t, c, key=dropout_key)
        unweighted_loss = jnp.mean(jnp.square((denoiser - x0)))
        return self.weighting(t) * unweighted_loss

    def score(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        sigma = self.sde.sigma(t)
        s = self.sde.s(t)
        diff = self.denoise(x / s, t, c, key=key) - x
        return diff / jnp.square(sigma)
