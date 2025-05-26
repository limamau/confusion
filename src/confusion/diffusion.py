from abc import abstractmethod
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from .networks import AbstractNetwork
from .sdes import AbstractSDE


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


class StandardDiffusionModel(AbstractDiffusionModel):
    network: AbstractNetwork

    def __init__(self, network: AbstractNetwork, sde: AbstractSDE):
        self.network = network
        super().__init__(sde)

    def score(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        return self.network(x, t, c, key=key) / self.sde.sigma(t)


class EDMDiffusionModel(AbstractDiffusionModel):
    sigma_data: float
    network: AbstractNetwork

    def __init__(self, network: AbstractNetwork, sde: AbstractSDE, sigma_data: float):
        self.network = network
        super().__init__(sde)
        self.sigma_data = sigma_data

    def score(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        sigma = self.sde.sigma(t)
        left_side = self.c_skip(sigma) * x
        eff_x = self.c_in(sigma) * x
        eff_t = self.c_noise(sigma) * sigma
        # note: on the EDM paper, they actually use sigma as the noise parameter scale
        # of the neural network F, but here I'm calling it eff_t for consistency with
        # the definition on AbstractNetwork and all it's sub-classes (inside networks/)
        right_side = self.c_out(sigma) * self.network(eff_x, eff_t, c, key=key)
        return left_side + right_side

    def c_skip(self, sigma):
        return self.sigma_data**2 / (self.sigma_data**2 + sigma**2)

    def c_out(self, sigma):
        return self.sigma_data * sigma / jnp.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / jnp.sqrt(self.sigma_data**2 + sigma**2)

    def c_noise(self, sigma):
        return 0.25 * jnp.log(sigma)
