from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..network import AbstractNetwork


# limamau: move this function to a "layers" to src/confusion/networks/points/layers.py
# and use both here and in the UNet
class GaussianFourierProjection(eqx.Module):
    gaussian: jax.Array

    def __init__(self, mapping_dim: int, scale: float = 10.0, *, key: Key):
        self.gaussian = jax.random.normal(key, (mapping_dim // 2,)) * scale

    def __call__(self, t: Array) -> Array:
        projection = t * self.gaussian * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(projection), jnp.cos(projection)], axis=-1).T


# limamau: write an instance of the equinox.nn.MultiheadAttention
# but now allowing for explicit masks that can will be used to encode
# the causal relations


class MultiLayerPerceptron(AbstractNetwork):
    temb: GaussianFourierProjection
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
        keys = jr.split(key, 6)

        if is_conditional:
            in_channels = num_variables + 2 + 1
        else:
            in_channels = num_variables + 2

        mapping_dim = 2
        scale = 1.0
        self.temb = GaussianFourierProjection(mapping_dim, scale, key=keys[0])
        self.in_linear = eqx.nn.Linear(in_channels, hidden_size, key=keys[1])
        self.hidden_linear1 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[2])
        self.hidden_linear2 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[3])
        self.hidden_linear3 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[4])
        self.out_linear = eqx.nn.Linear(hidden_size, num_variables, key=keys[5])

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
