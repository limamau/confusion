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
        self, x: Array, t: Array, c: Array | None, *, key: Key | None = None
    ) -> Array:
        raise NotImplementedError


class VPDiffusionModel(AbstractDiffusionModel):
    int_beta: Callable

    def __init__(
        self, network: AbstractNetwork, int_beta_fn: Callable, weights_fn: Callable
    ):
        self.network = network
        self.weights = weights_fn
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
        self, x: Array, t: Array, c: Array | None, *, key: Key | None = None
    ) -> Array:
        return self.network(x, t, c, key=key)
