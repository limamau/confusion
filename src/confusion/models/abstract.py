from abc import abstractmethod
from typing import Generic, Optional, Tuple, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from confusion.diffeqs.abstract import AbstractDiffEq
from confusion.guidance import AbstractGuidance

DiffEqType = TypeVar("DiffEqType", bound=AbstractDiffEq)


class AbstractModel(eqx.Module, Generic[DiffEqType]):
    diffeq: DiffEqType

    @property
    def sigma_max(self) -> Array:
        return self.diffeq.sigma(jnp.array(1.0))

    @abstractmethod
    def loss(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def probability_flow_ode(
        self,
        x: Array,
        t: Array,
        guidance: AbstractGuidance,
        pre_conds: Optional[Array] = None,
        post_conds: Optional[Array] = None,
    ) -> Array:
        """Returns the drift of the Probability Flow ODE."""
        raise NotImplementedError

    @abstractmethod
    def reverse_sde(
        self,
        x: Array,
        t: Array,
        guidance: AbstractGuidance,
        pre_conds: Optional[Array] = None,
        post_conds: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Returns the drift and diffusion of the Reverse SDE."""
        raise NotImplementedError
