from abc import abstractmethod
from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from .networks import AbstractNetwork


class AbstractDiffusionModel(eqx.Module):
    network: AbstractNetwork
    weights_fn: Callable
    t0: float
    t1: float

    @abstractmethod
    def s(self, t: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sigma(self, t: Array) -> Array:
        raise NotImplementedError

    # in practice, it is probably better to override this method
    # in order to provide a more efficient implementation
    def diffusion(self, t: Array) -> Array:
        s_t = self.s(t)
        sigma_t = self.sigma(t)
        sigma_t2 = jnp.square(sigma_t)
        sigma_t_dot = jnp.gradient(sigma_t2, t)
        s_t_dot = jnp.gradient(s_t, t)
        return sigma_t_dot * sigma_t - sigma_t2 * s_t_dot / s_t

    # again, it is probably better to override this method
    # in order to provide a more efficient implementation
    def drift(self, x: Array, t: Array) -> Array:
        s_t = self.s(t)
        s_t_dot = jnp.gradient(s_t, t)
        return s_t_dot / s_t * x

    def perturbation(self, x0: Array, t: Array, *, key: Key) -> Tuple[Array, Array]:
        return self.s(t) * x0, self.sigma(t)

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
    int_beta_fn: Callable

    def __init__(
        self,
        network: AbstractNetwork,
        int_beta_fn: Callable,
        t0: float,
        t1: float,
        weights_fn: Callable,
    ):
        self.network = network
        self.weights_fn = weights_fn
        self.t0 = t0
        self.t1 = t1
        self.int_beta_fn = int_beta_fn

    def s(self, t: Array) -> Array:
        return jnp.exp(-0.5 * self.int_beta_fn(t))

    def sigma(self, t: Array) -> Array:
        return jnp.sqrt(1 - jnp.exp(-self.int_beta_fn(t)))

    def diffusion(self, t: Array) -> Array:
        # get beta by derivating the integral
        _, beta = jax.jvp(self.int_beta_fn, (t,), (jnp.ones_like(t),))
        return jnp.sqrt(beta)

    def drift(self, x: Array, t: Array) -> Array:
        # same here
        _, beta = jax.jvp(self.int_beta_fn, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * x

    def score(
        self,
        x: Array,
        t: Array,
        c: Array | None,
        *,
        key: Optional[Key] = None,
    ) -> Array:
        return self.network(x, t, c, key=key)


class VarianceExploding(AbstractDiffusionModel):
    sigma_fn: Callable
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
        self.weights_fn = weights_fn
        self.t0 = t0
        self.t1 = t1
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        if is_approximate:
            self.sigma_fn = lambda t: sigma_min * jnp.pow((sigma_max / sigma_min), t)
        else:
            self.sigma_fn = lambda t: sigma_min * jnp.sqrt(
                jnp.exp(jnp.log(sigma_max / sigma_min) * t) - 1
            )

    def s(self, t: Array) -> Array:
        return jnp.ones_like(t)

    def sigma(self, t: Array) -> Array:
        return self.sigma_fn(t)

    def diffusion(self, t: Array) -> Array:
        log_ratio = jnp.sqrt(2 * jnp.log(self.sigma_max / self.sigma_min))
        return self.sigma(t) * log_ratio

    def drift(self, x: Array, t: Array) -> Array:
        return jnp.zeros_like(x)

    def score(
        self,
        x: Array,
        t: Array,
        c: Array | None,
        *,
        key: Key | None = None,
    ) -> Array:
        return self.network(x, t, c, key=key)
