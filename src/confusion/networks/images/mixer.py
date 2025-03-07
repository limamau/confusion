from typing import Optional, Tuple

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..networks import AbstractNaiveNetwork


class MixerBlock(eqx.Module):
    patch_mixer: eqx.nn.MLP
    hidden_mixer: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self,
        num_patches: int,
        hidden_size: int,
        mix_patch_size: int,
        mix_hidden_size: int,
        *,
        key: Key,
    ):
        tkey, ckey = jr.split(key, 2)
        self.patch_mixer = eqx.nn.MLP(
            num_patches, num_patches, mix_patch_size, depth=1, key=tkey
        )
        self.hidden_mixer = eqx.nn.MLP(
            hidden_size, hidden_size, mix_hidden_size, depth=1, key=ckey
        )
        self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))

    def __call__(self, x: Array) -> Array:
        x = x + jax.vmap(self.patch_mixer)(self.norm1(x))
        x = einops.rearrange(x, "c p -> p c")
        x = x + jax.vmap(self.hidden_mixer)(self.norm2(x))
        x = einops.rearrange(x, "p c -> c p")
        return x


class Mixer(AbstractNaiveNetwork):
    conv_in: eqx.nn.Conv2d
    conv_out: eqx.nn.ConvTranspose2d
    blocks: list
    norm: eqx.nn.LayerNorm
    t1: float

    def __init__(
        self,
        img_size: Tuple[int, int, int],
        patch_size: int,
        hidden_size: int,
        mix_patch_size: int,
        mix_hidden_size: int,
        num_blocks: int,
        t1: float,
        *,
        key: Key,
        is_conditional: bool = False,
    ):
        input_size, height, width = img_size
        assert (height % patch_size) == 0
        assert (width % patch_size) == 0
        num_patches = (height // patch_size) * (width // patch_size)
        inkey, outkey, *bkeys = jr.split(key, 2 + num_blocks)

        if is_conditional:
            in_channels = input_size + 2
        else:
            in_channels = input_size + 1

        self.conv_in = eqx.nn.Conv2d(
            in_channels, hidden_size, patch_size, stride=patch_size, key=inkey
        )
        self.conv_out = eqx.nn.ConvTranspose2d(
            hidden_size, input_size, patch_size, stride=patch_size, key=outkey
        )
        self.blocks = [
            MixerBlock(
                num_patches, hidden_size, mix_patch_size, mix_hidden_size, key=bkey
            )
            for bkey in bkeys
        ]
        self.norm = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.t1 = t1

    def __call__(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        t = jnp.array(t / self.t1)
        _, height, width = x.shape
        t = einops.repeat(t, "-> 1 h w", h=height, w=width)

        if c is not None:
            c = einops.repeat(c, "-> 1 h w", h=height, w=width)
            x = jnp.concatenate([x, t, c])
        else:
            x = jnp.concatenate([x, t])

        x = self.conv_in(x)
        _, patch_height, patch_width = x.shape
        x = einops.rearrange(x, "c h w -> c (h w)")
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = einops.rearrange(x, "c (h w) -> c h w", h=patch_height, w=patch_width)
        return self.conv_out(x)
