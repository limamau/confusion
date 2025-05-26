from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..layers import GaussianFourierProjection
from ..networks import AbstractNetwork


class ResBlock(eqx.Module):
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, hidden_size: int, *, key: Key):
        keys = jr.split(key, 3)
        self.linear1 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[0])
        self.linear2 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[1])
        self.norm1 = eqx.nn.GroupNorm(hidden_size // 4, hidden_size)
        self.norm2 = eqx.nn.GroupNorm(hidden_size // 4, hidden_size)

    def __call__(self, x: Array, skip: Array) -> Array:
        res = x
        x = self.norm1(x)
        x = jax.nn.swish(x)
        x = self.linear1(x)
        x = self.norm2(x)
        x = jax.nn.swish(x)
        x = self.linear2(x)
        return x + res + skip


class ResNet(AbstractNetwork):
    temb: GaussianFourierProjection
    in_linear: eqx.nn.Linear
    res_blocks: List[ResBlock]
    out_linear: eqx.nn.Linear

    def __init__(
        self,
        proj_size: int,
        proj_scale: float,
        num_variables: int,
        hidden_size: int,
        num_resblocks: int,
        *,
        key: Key,
        is_conditional: bool = False,
    ):
        keys = jr.split(key, 3 + num_resblocks)

        if is_conditional:
            in_channels = num_variables + proj_size // 2 * 2 + 1
        else:
            in_channels = num_variables + proj_size // 2 * 2

        self.temb = GaussianFourierProjection(proj_size, proj_scale, key=keys[0])
        self.in_linear = eqx.nn.Linear(in_channels, hidden_size, key=keys[1])
        self.out_linear = eqx.nn.Linear(hidden_size, num_variables, key=keys[2])
        self.res_blocks = [
            ResBlock(hidden_size, key=keys[i]) for i in range(3, 3 + num_resblocks)
        ]

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

        # resnet blocks
        x = self.in_linear(x)
        skip = x
        for block in self.res_blocks:
            x = block(x, skip)
        x = self.out_linear(x)

        return x
