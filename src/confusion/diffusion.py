from abc import abstractmethod
from typing import Callable, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from .networks import AbstractNetwork


class AbstractDiffusionModel(eqx.Module):
    network: AbstractNetwork
    weights2_fn: Callable

    @abstractmethod
    def s(self, t: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sigma(self, t: Array) -> Array:
        raise NotImplementedError

    # the following should be the inverse of sigma(t)
    # but I found no straight forward way to do this
    # automatically from sigma(t) using jax so it'll
    # stay as an abstract method which has to be implemented
    # in all instances of this class for now;
    # this is supposed to be used by the EDMStepSizeController
    @abstractmethod
    def t(self, sigma: Array) -> Array:
        raise NotImplementedError

    # in practice, it's probably better to override this method
    # in order to provide a more efficient implementation
    def diffusion(self, t: Array) -> Array:
        s_t = self.s(t)
        sigma_t = self.sigma(t)
        sigma_t2 = jnp.square(sigma_t)
        sigma_t_dot = jnp.gradient(sigma_t2, t)
        s_t_dot = jnp.gradient(s_t, t)
        return jnp.sqrt(sigma_t_dot * sigma_t - sigma_t2 * s_t_dot / s_t)

    # again, it's probably better to override this method
    # in order to provide a more efficient implementation
    def drift(self, x: Array, t: Array) -> Array:
        s_t = self.s(t)
        s_t_dot = jnp.gradient(s_t, t)
        return s_t_dot / s_t * x

    def perturbation(self, x0: Array, t: Array) -> Tuple[Array, Array]:
        return self.s(t) * x0, self.sigma(t)

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


class VariancePreserving(AbstractDiffusionModel):
    beta_min_bar: float
    beta_max_bar: float

    def __init__(
        self,
        network: AbstractNetwork,
        weights2_fn: Callable,
        beta_min_bar: float,
        beta_max_bar: float,
    ):
        self.network = network
        self.weights2_fn = weights2_fn
        self.beta_min_bar = beta_min_bar
        self.beta_max_bar = beta_max_bar

    def s(self, t: Array) -> Array:
        return jnp.exp(
            -0.25 * t**2 * (self.beta_max_bar - self.beta_min_bar)
            - 0.5 * t * self.beta_min_bar
        )

    def sigma(self, t: Array) -> Array:
        return jnp.sqrt(
            1
            - jnp.exp(
                -0.5 * t**2 * (self.beta_max_bar - self.beta_min_bar)
                - t * self.beta_min_bar
            )
        )

    def t(self, sigma: Array) -> Array:
        return (
            jnp.sqrt(
                self.beta_min_bar**2
                - 2 * (self.beta_max_bar - self.beta_min_bar) * jnp.log(1 - sigma**2)
            )
            - self.beta_min_bar
        ) / (self.beta_max_bar - self.beta_min_bar)

    def diffusion(self, t: Array) -> Array:
        return jnp.sqrt(self.beta_max_bar + t * (self.beta_max_bar - self.beta_min_bar))

    def drift(self, x: Array, t: Array) -> Array:
        return -0.5 * (self.beta_min_bar + t * (self.beta_max_bar - self.beta_min_bar))

    def score(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        return self.network(x, t, c, key=key)


class VarianceExploding(AbstractDiffusionModel):
    sigma_min: float
    sigma_max: float
    sigma_fn: Callable
    t_fn: Callable

    def __init__(
        self,
        network: AbstractNetwork,
        weights2_fn: Callable,
        sigma_min: float,
        sigma_max: float,
        is_approximate: bool = True,
    ):
        self.network = network
        self.weights2_fn = weights2_fn
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # this is a good approximation for sigma_max >> sigma_min
        if is_approximate:
            self.sigma_fn = lambda t: sigma_min * jnp.pow((sigma_max / sigma_min), t)
            self.t_fn = lambda sigma: jnp.log(sigma / sigma_min) / jnp.log(
                sigma_max / sigma_min
            )
        else:
            self.sigma_fn = lambda t: sigma_min * jnp.sqrt(
                jnp.pow(sigma_max / sigma_min, 2 * t) - 1
            )  # limamau: somehow using t instead of 2t leads to better results...
            self.t_fn = lambda sigma: jnp.log(sigma**2 / sigma_min**2 + 1) / (
                2 * jnp.log(sigma_max / sigma_min)
            )

    def s(self, t: Array) -> Array:
        return jnp.ones_like(t)

    def sigma(self, t: Array) -> Array:
        return self.sigma_fn(t)

    def t(self, sigma: Array) -> Array:
        return self.t_fn(sigma)

    def diffusion(self, t: Array) -> Array:
        log_ratio = jnp.sqrt(2 * jnp.log(self.sigma_max / self.sigma_min))
        return self.sigma(t) * log_ratio

    def drift(self, x: Array, t: Array) -> Array:
        return jnp.zeros_like(x)

    def score(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        return self.network(x, t, c, key=key)
