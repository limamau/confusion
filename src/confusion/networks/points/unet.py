from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from confusion.networks import AbstractNetwork

from ..layers import GaussianFourierProjection


class ResBlock(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    dropout: eqx.nn.Dropout

    def __init__(self, hidden_size: int, dropout_rate: float, *, key: Key):
        keys = jr.split(key, 3)
        self.linear1 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[0])
        self.linear2 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[1])
        self.norm1 = eqx.nn.GroupNorm(hidden_size // 4, hidden_size)
        self.norm2 = eqx.nn.GroupNorm(hidden_size // 4, hidden_size)
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, x: Array, key: Optional[Key]) -> Array:
        residual = x
        x = self.norm1(x)
        x = jax.nn.swish(x)
        x = self.linear1(x)
        x = self.norm2(x)
        x = jax.nn.swish(x)
        x = self.linear2(x)
        if key is not None:
            x = self.dropout(x, key=key)
        return x + residual


class DownBlock(eqx.Module):
    resblocks: List[ResBlock]
    downsample: eqx.nn.Linear

    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_resblocks: int,
        dropout_rate: float,
        *,
        key: Key,
    ):
        keys = jr.split(key, num_resblocks + 1)
        self.resblocks = [
            ResBlock(in_size, dropout_rate=dropout_rate, key=keys[i])
            for i in range(num_resblocks)
        ]
        self.downsample = eqx.nn.Linear(in_size, out_size, key=keys[-1])

    def __call__(self, x: Array, key: Optional[Key]) -> Tuple[Array, Array]:
        skip = x
        for block in self.resblocks:
            x = block(x, key=key)
        skip = x
        x = self.downsample(x)
        x = jax.nn.swish(x)
        return x, skip


class UpBlock(eqx.Module):
    resblocks: List[ResBlock]
    upsample: eqx.nn.Linear

    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_resblocks: int,
        dropout_rate: float,
        *,
        key: Key,
    ):
        keys = jr.split(key, num_resblocks + 1)
        self.resblocks = [
            ResBlock(out_size, dropout_rate=dropout_rate, key=keys[i])
            for i in range(num_resblocks)
        ]
        self.upsample = eqx.nn.Linear(in_size, out_size, key=keys[-1])

    def __call__(self, x: Array, skip: Array, key: Optional[Key]) -> Array:
        x = self.upsample(x)
        x = jax.nn.swish(x)
        x = x + skip
        for block in self.resblocks:
            x = block(x, key=key)
        return x


class MiddleBlock(eqx.Module):
    resblocks: List[ResBlock]
    attention: eqx.nn.MultiheadAttention
    in_linear: eqx.nn.Linear
    out_linear: eqx.nn.Linear

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qkv_size: int,
        dropout_rate: float,
        *,
        key: Key,
    ):
        keys = jr.split(key, 5)
        self.resblocks = [
            ResBlock(hidden_size, dropout_rate=dropout_rate, key=keys[0]),
            ResBlock(hidden_size, dropout_rate=dropout_rate, key=keys[1]),
        ]
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=1,
            query_size=qkv_size,
            key=keys[2],
        )
        self.in_linear = eqx.nn.Linear(hidden_size, qkv_size * 3, key=keys[3])
        self.out_linear = eqx.nn.Linear(qkv_size, hidden_size, key=keys[4])

    def __call__(self, x: Array, *, key: Optional[Key] = None) -> Array:
        x = self.resblocks[0](x, key=key)
        # x = x[:, None]
        # qkv = self.in_linear(x)
        # q, k, v = jnp.split(qkv, 3)
        # x = self.attention(q, k, v)
        # x = self.out_linear(x)
        # x = x[:, 0]
        x = self.resblocks[1](x, key=key)
        return x


class PointUNet(AbstractNetwork):
    temb: GaussianFourierProjection
    in_linear: eqx.nn.Linear

    down1: DownBlock
    down2: DownBlock

    middle: MiddleBlock

    up2: UpBlock
    up1: UpBlock

    out_linear: eqx.nn.Linear

    def __init__(
        self,
        proj_size: int,
        proj_scale: float,
        num_variables: int,
        hidden_size: int,
        num_heads: int,
        qkv_size: int,
        dropout_rate: float,
        *,
        key: Key,
        is_conditional: bool = False,
    ):
        keys = jr.split(key, 8)

        if is_conditional:
            in_channels = num_variables + proj_size // 2 * 2 + 1
        else:
            in_channels = num_variables + proj_size // 2 * 2

        # (SDE) time embedding
        self.temb = GaussianFourierProjection(proj_size, proj_scale, key=keys[0])
        self.in_linear = eqx.nn.Linear(in_channels, hidden_size, key=keys[1])

        # sizes for each level
        h1_size = hidden_size
        h2_size = hidden_size * 2
        h3_size = hidden_size * 4

        # downsampling
        self.down1 = DownBlock(
            h1_size, h2_size, num_resblocks=2, dropout_rate=dropout_rate, key=keys[2]
        )
        self.down2 = DownBlock(
            h2_size, h3_size, num_resblocks=2, dropout_rate=dropout_rate, key=keys[3]
        )

        # middle block
        self.middle = MiddleBlock(
            h3_size, num_heads, qkv_size, dropout_rate=dropout_rate, key=keys[4]
        )

        # upsampling
        self.up2 = UpBlock(
            h3_size, h2_size, num_resblocks=2, dropout_rate=dropout_rate, key=keys[5]
        )
        self.up1 = UpBlock(
            h2_size, h1_size, num_resblocks=2, dropout_rate=dropout_rate, key=keys[6]
        )

        # output
        self.out_linear = eqx.nn.Linear(h1_size, num_variables, key=keys[7])

    def __call__(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        if key is not None:
            key1, key2, key3, key4, key5 = jr.split(key, 5)
        else:
            key1 = key2 = key3 = key4 = key5 = None

        # time embedding (Fourier)
        t = self.temb(t)

        # concatenate inputs
        if c is not None:
            c = jnp.expand_dims(c, axis=0) if c.ndim == 0 else c
            x = jnp.concatenate([x, t, c])
        else:
            x = jnp.concatenate([x, t])

        # initial projection
        x = self.in_linear(x)
        x = jax.nn.swish(x)

        # downsampling (store skip connections)
        x1, skip1 = self.down1(x, key=key1)
        x2, skip2 = self.down2(x1, key=key2)

        # middle block with attention
        x = self.middle(x2, key=key3)

        # upsampling (use skip connections)
        x = self.up2(x, skip2, key=key4)
        x = self.up1(x, skip1, key=key5)

        # output projection
        x = self.out_linear(x)

        return x
