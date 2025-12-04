from abc import abstractmethod
from typing import Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from confusion.diffeqs.odes import AbstractODE
from confusion.guidance import AbstractGuidance
from confusion.models import AbstractModel
from confusion.networks import AbstractNetwork


class AbstractDriftModel(AbstractModel[AbstractODE]):
    network: AbstractNetwork

    def __init__(self, network: AbstractNetwork, ode: AbstractODE):
        self.network = network
        self.diffeq = ode

    @abstractmethod
    def drift(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
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
        return self.drift(x, t, pre_conds)

    def reverse_sde(
        self,
        x: Array,
        t: Array,
        guidance: AbstractGuidance,
        pre_conds: Optional[Array] = None,
        post_conds: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        raise NotImplementedError("Drift models do not support SDE sampling.")

    @abstractmethod
    def loss(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        raise NotImplementedError


class StandardFlowMatching(AbstractDriftModel):
    def __init__(
        self,
        network: AbstractNetwork,
        ode: AbstractODE,
    ):
        super().__init__(network, ode)

    def loss(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        key: Key,
    ) -> Array:
        x0 = x
        dropout_key, sample_key = jr.split(key)
        x1 = jr.normal(sample_key, x0.shape)
        x_t = self.diffeq.perturbation(x0, t, x1)
        u_t = self.diffeq.vector_field(x_t, t, x0)
        v_t = self.network(x_t, t, c, key=dropout_key)
        return jnp.mean(jnp.square(v_t - u_t))

    def drift(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        return self.network(x, t, c, key=key)
