from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..layers import GaussianFourierProjection
from ..networks import AbstractNetwork


class MultiLayerPerceptron(AbstractNetwork):
    temb: GaussianFourierProjection
    in_linear: eqx.nn.Linear
    hidden_linear1: eqx.nn.Linear
    hidden_linear2: eqx.nn.Linear
    hidden_linear3: eqx.nn.Linear
    out_linear: eqx.nn.Linear

    def __init__(
        self,
        proj_size: int,
        proj_scale: float,
        num_variables: int,
        hidden_size: int,
        *,
        key: Key,
        is_conditional: bool = False,
    ):
        keys = jr.split(key, 6)

        if is_conditional:
            in_channels = num_variables + proj_size // 2 * 2 + 1
        else:
            in_channels = num_variables + proj_size // 2 * 2

        self.temb = GaussianFourierProjection(proj_size, proj_scale, key=keys[0])
        self.in_linear = eqx.nn.Linear(in_channels, hidden_size, key=keys[1])
        self.hidden_linear1 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[2])
        self.hidden_linear2 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[3])
        self.hidden_linear3 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[4])
        self.out_linear = eqx.nn.Linear(hidden_size, num_variables, key=keys[5])

    def __call__(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        # time embedding (Fourier)
        t = self.temb(t)

        # concatenate inputs
        if c is not None:
            c = jnp.expand_dims(c, axis=0) if c.ndim == 0 else c
            x = jnp.concatenate([x, t, c])
        else:
            x = jnp.concatenate([x, t])

        # MLP
        x = self.in_linear(x)
        x = jax.nn.swish(x)
        x = self.hidden_linear1(x)
        x = jax.nn.swish(x)
        x = self.hidden_linear2(x)
        x = jax.nn.swish(x)
        x = self.hidden_linear3(x)
        x = jax.nn.swish(x)
        x = self.out_linear(x)

        return x
