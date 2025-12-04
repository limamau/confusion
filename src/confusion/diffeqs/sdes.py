from abc import abstractmethod
from typing import Callable, Tuple

import jax.numpy as jnp
from jaxtyping import Array

from confusion.diffeqs.abstract import AbstractDiffEq


class AbstractSDE(AbstractDiffEq):
    @abstractmethod
    def mu(self, t: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sigma(self, t: Array) -> Array:
        raise NotImplementedError

    # in practice, it's probably better to override this method
    # in order to provide a more efficient implementation
    def diffusion(self, t: Array) -> Array:
        mu_t = self.mu(t)
        sigma_t = self.sigma(t)
        sigma_t2 = jnp.square(sigma_t)
        sigma_t_dot = jnp.gradient(sigma_t2, t)
        mu_t_dot = jnp.gradient(mu_t, t)
        return jnp.sqrt(sigma_t_dot * sigma_t - sigma_t2 * mu_t_dot / mu_t)

    # again, it's probably better to override this method
    # in order to provide a more efficient implementation
    def drift(self, x: Array, t: Array) -> Array:
        mu_t = self.mu(t)
        mu_t_dot = jnp.gradient(mu_t, t)
        return mu_t_dot / mu_t * x

    def perturbation(self, x0: Array, t: Array) -> Tuple[Array, Array]:
        return self.mu(t) * x0, self.sigma(t)


class SubVariancePreserving(AbstractSDE):
    beta_min: float
    beta_max: float

    def __init__(
        self,
        beta_min: float,
        beta_max: float,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _alpha(self, t: Array) -> Array:
        return 0.5 * t**2 * (self.beta_max - self.beta_min) + t * self.beta_min

    def _beta(self, t: Array) -> Array:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def mu(self, t: Array) -> Array:
        return jnp.exp(-0.5 * self._alpha(t))

    def sigma(self, t: Array) -> Array:
        return 1 - jnp.exp(-self._alpha(t))

    def diffusion(self, t: Array) -> Array:
        return jnp.sqrt(self._beta(t) * (1 - jnp.exp(-2 * self._alpha(t))))

    def drift(self, x: Array, t: Array) -> Array:
        return -0.5 * self._beta(t) * x


class VariancePreserving(AbstractSDE):
    beta_min: float
    beta_max: float

    def __init__(
        self,
        beta_min: float,
        beta_max: float,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _alpha(self, t: Array) -> Array:
        return 0.5 * (t**2) * (self.beta_max - self.beta_min) + t * self.beta_min

    def _beta(self, t: Array) -> Array:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def mu(self, t: Array) -> Array:
        return jnp.exp(-0.5 * self._alpha(t))

    def sigma(self, t: Array) -> Array:
        return jnp.sqrt(1 - jnp.exp(-self._alpha(t)))

    def diffusion(self, t: Array) -> Array:
        return jnp.sqrt(self._beta(t))

    def drift(self, x: Array, t: Array) -> Array:
        return -0.5 * self._beta(t) * x


class VarianceExploding(AbstractSDE):
    sigma_min: float
    sigma_max: float
    sigma_fn: Callable
    t_fn: Callable

    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        is_approximate: bool = True,
    ):
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
            )
            self.t_fn = lambda sigma: jnp.log(sigma**2 / sigma_min**2 + 1) / (
                2 * jnp.log(sigma_max / sigma_min)
            )

    def mu(self, t: Array) -> Array:
        return jnp.ones_like(t)

    def sigma(self, t: Array) -> Array:
        return self.sigma_fn(t)

    def diffusion(self, t: Array) -> Array:
        log_ratio = jnp.sqrt(2 * jnp.log(self.sigma_max / self.sigma_min))
        return self.sigma(t) * log_ratio

    def drift(self, x: Array, t: Array) -> Array:
        return jnp.zeros_like(x)
