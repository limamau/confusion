from abc import abstractmethod
from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .networks import AbstractNetwork


class AbstractDiffusionModel(eqx.Module):
    network: AbstractNetwork
    weights: Callable
    t0: float
    t1: float

    @abstractmethod
    def diffusion(self, t: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def drift(self, x: Array, t: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def perturbation(
        self, x0: Array, t: Array, *, key: Key
    ) -> Tuple[Array, Array, Array]:
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        x: Array,
        t: Array,
        c: Array | None,
        *,
        key: Key | None = None,
    ) -> Array:
        raise NotImplementedError


class VariancePreserving(AbstractDiffusionModel):
    int_beta: Callable

    def __init__(
        self,
        network: AbstractNetwork,
        int_beta_fn: Callable,
        t0: float,
        t1: float,
        weights_fn: Callable,
    ):
        self.network = network
        self.weights = weights_fn
        self.t0 = t0
        self.t1 = t1
        self.int_beta = int_beta_fn

    def diffusion(self, t: Array) -> Array:
        # get beta by derivating the integral
        _, beta = jax.jvp(self.int_beta, (t,), (jnp.ones_like(t),))
        return jnp.sqrt(beta)

    def drift(self, x: Array, t: Array) -> Array:
        # same here
        _, beta = jax.jvp(self.int_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * x

    def perturbation(
        self, x0: Array, t: Array, *, key: Key
    ) -> Tuple[Array, Array, Array]:
        mean = x0 * jnp.exp(-0.5 * self.int_beta(t))
        std = jnp.sqrt(jnp.maximum(1 - jnp.exp(-self.int_beta(t)), 1e-5))
        noise = jr.normal(key, x0.shape)
        x = mean + std * noise
        return x, noise, std

    def score(
        self,
        x: Array,
        t: Array,
        c: Array | None,
        *,
        key: Key | None = None,
    ) -> Array:
        return self.network(x, t, c, key=key)


class VarianceExploding(AbstractDiffusionModel):
    sigma: Callable
    sigma_min: float
    sigma_max: float

    def __init__(
        self,
        network: AbstractNetwork,
        weights_fn: Callable,
        t0: float,
        t1: float,
        sigma_min: float,
        sigma_max: float,
        is_approximate: bool = True,
    ):
        self.network = network
        self.weights = weights_fn
        self.t0 = t0
        self.t1 = t1
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        if is_approximate:
            self.sigma = lambda t: sigma_min * jnp.pow((sigma_max / sigma_min), t)
        else:
            self.sigma = lambda t: sigma_min * jnp.sqrt(
                jnp.exp(jnp.log(sigma_max / sigma_min) * t) - 1
            )

    def diffusion(self, t: Array) -> Array:
        log_ratio = jnp.sqrt(2 * jnp.log(self.sigma_max / self.sigma_min))
        return self.sigma(t) * log_ratio

    def drift(self, x: Array, t: Array) -> Array:
        return jnp.zeros_like(x)

    def perturbation(
        self, x0: Array, t: Array, *, key: Key
    ) -> Tuple[Array, Array, Array]:
        noise = jr.normal(key, x0.shape)
        std = self.sigma(t)
        x = x0 + std * noise
        return x, noise, std

    def score(
        self,
        x: Array,
        t: Array,
        c: Array | None,
        *,
        key: Key | None = None,
    ) -> Array:
        return self.network(x, t, c, key=key)
