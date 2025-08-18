import jax.numpy as jnp
from jaxtyping import Array

from .sdes import AbstractSDE


class AbstractWeighting:
    def __call__(self, t: Array) -> Array:
        raise NotImplementedError


class NoWeighting(AbstractWeighting):
    def __init__(self):
        pass

    def __call__(self, t: Array) -> Array:
        return jnp.ones_like(t)


class DirectWeighting(AbstractWeighting):
    def __init__(self, sde: AbstractSDE):
        self.sde = sde

    def __call__(self, t: Array) -> Array:
        return self.sde.sigma(t)


class SquaredWeighting(AbstractWeighting):
    def __init__(self, sde: AbstractSDE):
        self.sde = sde

    def __call__(self, t: Array) -> Array:
        return jnp.square(self.sde.sigma(t))


class InverseWeighting(AbstractWeighting):
    def __init__(self, sde: AbstractSDE):
        self.sde = sde

    def __call__(self, t: Array) -> Array:
        return 1 / self.sde.sigma(t) + 1


class InverseSquaredWeighting(AbstractWeighting):
    def __init__(self, sde: AbstractSDE):
        self.sde = sde

    def __call__(self, t: Array) -> Array:
        return 1 / jnp.square(self.sde.sigma(t)) + 1


class DenoiserWeighting(AbstractWeighting):
    def __init__(self, sde: AbstractSDE, sigma_data: float):
        self.sde = sde
        self.sigma_data = sigma_data

    def __call__(self, t: Array) -> Array:
        sigma = self.sde.sigma(t)
        num = jnp.sqrt(sigma**2 + self.sigma_data**2)
        den = sigma * self.sigma_data
        return num / den
