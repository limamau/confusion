from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..network import AbstractNetwork


class FourierTimeEmbedding(eqx.Module):
    def __call__(self, t: Array) -> Array:
        t = jnp.expand_dims(t, axis=-1)
        return jnp.concatenate(
            [jnp.cos(2 * jnp.pi * t), jnp.sin(2 * jnp.pi * t)], axis=-1
        )


class MultiLayerPerceptron(AbstractNetwork):
    temb: FourierTimeEmbedding
    in_linear: eqx.nn.Linear
    hidden_linear1: eqx.nn.Linear
    hidden_linear2: eqx.nn.Linear
    hidden_linear3: eqx.nn.Linear
    out_linear: eqx.nn.Linear

    def __init__(
        self,
        num_variables: int,
        hidden_size: int,
        t1: float,
        *,
        key: Key,
        is_conditional=True,
    ):
        keys = jr.split(key, 5)

        if is_conditional:
            in_channels = num_variables + 2 + 1
        else:
            in_channels = num_variables + 2

        self.temb = FourierTimeEmbedding()
        self.in_linear = eqx.nn.Linear(in_channels, hidden_size, key=keys[0])
        self.hidden_linear1 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[1])
        self.hidden_linear2 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[2])
        self.hidden_linear3 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[3])
        self.out_linear = eqx.nn.Linear(hidden_size, num_variables, key=keys[4])

    def __call__(
        self,
        y: Array,
        t: Array,
        c: Array | None,
        *,
        key: Optional[Key] = None,
    ) -> Array:
        # time embedding (Fourier)
        t = self.temb(t)

        # concatenate inputs
        if c is not None:
            c = jnp.expand_dims(c, axis=0) if c.ndim == 0 else c
            y = jnp.concatenate([y, t, c])
        else:
            y = jnp.concatenate([y, t])

        # MLP
        y = self.in_linear(y)
        y = jax.nn.swish(y)
        y = self.hidden_linear1(y)
        y = jax.nn.swish(y)
        y = self.hidden_linear2(y)
        y = jax.nn.swish(y)
        y = self.hidden_linear3(y)
        y = jax.nn.swish(y)
        y = self.out_linear(y)

        return y
