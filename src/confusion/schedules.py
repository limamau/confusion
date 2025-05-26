from abc import abstractmethod
from typing import Optional

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key


class AbstractTimeSchedule:
    use_low_discrepancy: bool

    def __init__(self, use_low_discrepancy: bool = True):
        self.use_low_discrepancy = use_low_discrepancy

    # the idea here is to get a deterministic array of times when key is None (e.g., during sampling)
    # and to get a random one (with low discrepancy or not) when key is not None (e.g., during training)
    @abstractmethod
    def __call__(
        self, t0: float, t1: float, size: int, key: Optional[Key] = None
    ) -> Array:
        raise NotImplementedError


class LinearTimeSchedule(AbstractTimeSchedule):
    def __init__(self, use_low_discrepancy: bool = True):
        super().__init__(use_low_discrepancy)

    def __call__(
        self, t0: float, t1: float, size: int, key: Optional[Key] = None
    ) -> Array:
        if key is not None:
            if self.use_low_discrepancy:
                u = jr.uniform(key, (size,), maxval=1.0 / size)
                u += jnp.arange(size) / size
            else:
                u = jr.uniform(key, (size,), maxval=1.0)
        else:
            u = jnp.linspace(0.0, 1.0, size)

        return t0 + u * (t1 - t0)


# if exponent = 1.0, this is equivalent to LinearTimeSchedule
class PowerTimeSchedule(AbstractTimeSchedule):
    def __init__(self, use_low_discrepancy: bool = True, exponent: float = 12.0):
        super().__init__(use_low_discrepancy)
        self.exponent = exponent

    def __call__(
        self, t0: float, t1: float, size: int, key: Optional[Key] = None
    ) -> Array:
        if key is not None:
            if self.use_low_discrepancy:
                u = jr.uniform(key, (size,), maxval=1.0 / size)
                u += jnp.arange(size) / size
            else:
                u = jr.uniform(key, (size,), maxval=1.0)
        else:
            u = jnp.linspace(0.0, 1.0, size)

        return t0 + u**self.exponent * (t1 - t0)


# I'm leaving this here...
# class EDMTimeSchedule(AbstractTimeSchedule):
#     sde: AbstractSDE
#     rho: float
#     tol: float

#     def __init__(
#         self,
#         sde: AbstractSDE,
#         rho: float = 7.0,
#         tol: float = 1e-3,
#         low_discrepancy: bool = True,
#     ):
#         super().__init__(low_discrepancy)
#         self.sde = sde
#         self.rho = rho
#         self.tol = tol

#     def __call__(
#         self,
#         t0: float,
#         t1: float,
#         size: int,
#         key: Optional[Key] = None,
#     ):
#         sigma_max = self.sde.sigma(jnp.array([t1]))
#         sigma_min = self.sde.sigma(jnp.array([t0]))

#         sigmas = jnp.power(
#             jnp.power(sigma_max, (1 / self.rho))
#             + jnp.arange(size)
#             / (size - 1)
#             * (
#                 jnp.power(sigma_min, (1 / self.rho))
#                 - jnp.power(sigma_max, (1 / self.rho))
#             ),
#             self.rho,
#         )

#         t_array = jnp.array([self.sde.t(sigma_i) for sigma_i in sigmas])[::-1]

#         # bypass approximations on tails
#         assert jnp.abs(t_array[0] - t1) < self.tol, (
#             f"t_array[0] ({t_array[0]}) is not close to t1 ({t1})"
#         )
#         assert jnp.abs(t_array[-1] - t0) < self.tol, (
#             f"t_array[-1] ({t_array[-1]}) is not close to t0 ({t0})"
#         )
#         t_array = t_array.at[0].set(t1)
#         t_array = t_array.at[-1].set(t0)

#         return t_array
