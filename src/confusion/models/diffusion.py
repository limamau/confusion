from abc import abstractmethod
from typing import Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from confusion.diffeqs.sdes import AbstractSDE
from confusion.guidance import AbstractGuidance
from confusion.models import AbstractModel
from confusion.networks import AbstractNetwork
from confusion.weighting import AbstractWeighting


class AbstractDiffusionModel(AbstractModel[AbstractSDE]):
    def __init__(self, sde: AbstractSDE):
        self.diffeq = sde

    @abstractmethod
    def score(
        self,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        key: Optional[Key] = None,
    ) -> Array:
        raise NotImplementedError

    def probability_flow_ode(
        self,
        x: Array,
        t: Array,
        guidance: AbstractGuidance,
        pre_conds: Optional[Array] = None,
        post_conds: Optional[Array] = None,
    ) -> Array:
        f = self.diffeq.drift(x, t)
        g2 = jnp.square(self.diffeq.diffusion(t))

        def score_fn(x, t):
            return self.score(x, t, pre_conds, key=None)

        score = guidance.apply_on_score(
            score_fn, self.diffeq, x, t, pre_conds, post_conds, key=None
        )
        return f - 0.5 * g2 * score

    def reverse_sde(
        self,
        x: Array,
        t: Array,
        guidance: AbstractGuidance,
        pre_conds: Optional[Array] = None,
        post_conds: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        f = self.diffeq.drift(x, t)
        g = self.diffeq.diffusion(t)
        g2 = jnp.square(g)

        def score_fn(x, t):
            return self.score(x, t, pre_conds, key=None)

        score = guidance.apply_on_score(
            score_fn, self.diffeq, x, t, pre_conds, post_conds, key=None
        )
        return f - g2 * score, g

    @abstractmethod
    def loss(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        raise NotImplementedError


class StandardDiffusionModel(AbstractDiffusionModel):
    network: AbstractNetwork
    weighting: AbstractWeighting

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
        x: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        x0 = x

        noise_key, dropout_key = jr.split(key)
        mean, std = self.diffeq.perturbation(x0, t)
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
        pre_conds: Optional[Array],
        key: Optional[Key] = None,
    ) -> Array:
        c = pre_conds
        return self.network(x, t, c, key=key) / self.diffeq.sigma(t)


class DenoiserDiffusionModel(AbstractDiffusionModel):
    network: AbstractNetwork
    weighting: AbstractWeighting
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
        pre_conds: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        c = pre_conds
        sigma = self.diffeq.sigma(t)
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
        x: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        x0 = x

        noise_key, dropout_key = jr.split(key)
        # limamau: take out the perturbation method as this is only used in the standard diffusion model
        # limamau: also go from s to mu in notation/method
        std = self.diffeq.sigma(t)
        standard_noise = jr.normal(key, x0.shape)
        x = x0 + std * standard_noise
        denoiser = self.denoise(x, t, c, key=dropout_key)
        unweighted_loss = jnp.mean(jnp.square((denoiser - x0)))
        return self.weighting(t) * unweighted_loss

    def score(
        self,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        key: Optional[Key] = None,
    ) -> Array:
        c = pre_conds
        sigma = self.diffeq.sigma(t)
        mu = self.diffeq.mu(t)
        diff = self.denoise(x / mu, t, c, key=key) - x
        return diff / jnp.square(sigma)
