from collections.abc import Callable
from typing import List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Key

from ..layers import GaussianFourierProjection
from ..networks import AbstractNetwork


class LinearTimeSelfAttention(eqx.Module):
    group_norm: eqx.nn.GroupNorm
    heads: int
    to_qkv: eqx.nn.Conv2d
    to_out: eqx.nn.Conv2d

    def __init__(
        self,
        dim: int,
        key: Key,
        heads: int = 4,
        dim_head: int = 32,
    ):
        keys = jax.random.split(key, 2)
        self.group_norm = eqx.nn.GroupNorm(min(dim // 4, 32), dim)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = eqx.nn.Conv2d(dim, hidden_dim * 3, 1, key=keys[0])
        self.to_out = eqx.nn.Conv2d(hidden_dim, dim, 1, key=keys[1])

    def __call__(self, x: Array) -> Array:
        c, h, w = x.shape
        x = self.group_norm(x)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "(qkv heads c) h w -> qkv heads c (h w)", heads=self.heads, qkv=3
        )
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum("hdn,hen->hde", k, v)
        out = jnp.einsum("hde,hdn->hen", context, q)
        out = rearrange(
            out, "heads c (h w) -> (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


def upsample_2d(x: Array, factor: int = 2) -> Array:
    C, H, W = x.shape
    x = jnp.reshape(x, [C, H, 1, W, 1])
    x = jnp.tile(x, [1, 1, factor, 1, factor])
    return jnp.reshape(x, [C, H * factor, W * factor])


def downsample_2d(x: Array, factor: int = 2) -> Array:
    C, H, W = x.shape
    x = jnp.reshape(x, [C, H // factor, factor, W // factor, factor])
    return jnp.mean(x, axis=[2, 4])


def exact_zip(*args):
    _len = len(args[0])
    for arg in args:
        assert len(arg) == _len
    return zip(*args)


def key_split_allowing_none(key: Key | None) -> Tuple[Key | None, Key | None]:
    if key is None:
        return key, None
    else:
        k1, k2 = jr.split(key)
        return k1, k2


class Residual(eqx.Module):
    fn: LinearTimeSelfAttention

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        return self.fn(x, *args, **kwargs) + x


class ResnetBlock(eqx.Module):
    dim_out: int
    is_biggan: bool
    up: bool
    down: bool
    dropout_rate: float
    time_emb_dim: int
    t_mlp_layers: List[Union[Callable, eqx.nn.Linear]]
    c_mlp_layers: List[Union[Callable, eqx.nn.Linear]]
    scaling: Union[None, Callable, eqx.nn.ConvTranspose2d, eqx.nn.Conv2d]
    block1_groupnorm: eqx.nn.GroupNorm
    block1_conv: eqx.nn.Conv2d
    block2_layers: List[
        Union[eqx.nn.GroupNorm, eqx.nn.Dropout, eqx.nn.Conv2d, Callable]
    ]
    res_conv: eqx.nn.Conv2d
    attn: Optional[Residual]

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        is_biggan: bool,
        up: bool,
        down: bool,
        time_emb_dim: int,
        cond_emb_dim: int,
        dropout_rate: float,
        is_attn: bool,
        heads: int,
        dim_head: int,
        *,
        key: Key,
        is_conditional: bool = False,
    ):
        keys = jax.random.split(key, 7)
        self.dim_out = dim_out
        self.is_biggan = is_biggan
        self.up = up
        self.down = down
        self.dropout_rate = dropout_rate
        self.time_emb_dim = time_emb_dim

        self.t_mlp_layers = [
            jax.nn.silu,
            eqx.nn.Linear(time_emb_dim, dim_out, key=keys[0]),
        ]

        if is_conditional:
            self.c_mlp_layers = [
                jax.nn.silu,
                eqx.nn.Linear(cond_emb_dim, dim_out, key=keys[0]),
            ]
        else:
            self.c_mlp_layers = []

        self.block1_groupnorm = eqx.nn.GroupNorm(min(dim_in // 4, 32), dim_in)
        self.block1_conv = eqx.nn.Conv2d(dim_in, dim_out, 3, padding=1, key=keys[1])
        self.block2_layers = [
            eqx.nn.GroupNorm(min(dim_out // 4, 32), dim_out),
            jax.nn.silu,
            eqx.nn.Dropout(dropout_rate),
            eqx.nn.Conv2d(dim_out, dim_out, 3, padding=1, key=keys[2]),
        ]

        assert not self.up or not self.down

        if is_biggan:
            if self.up:
                self.scaling = upsample_2d
            elif self.down:
                self.scaling = downsample_2d
            else:
                self.scaling = None
        else:
            if self.up:
                self.scaling = eqx.nn.ConvTranspose2d(
                    dim_in,
                    dim_in,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    key=keys[3],
                )
            elif self.down:
                self.scaling = eqx.nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    key=keys[4],
                )
            else:
                self.scaling = None
        # For DDPM Yang use their own custom layer called NIN, which is
        # equivalent to a 1x1 conv
        self.res_conv = eqx.nn.Conv2d(dim_in, dim_out, kernel_size=1, key=keys[5])

        if is_attn:
            self.attn = Residual(
                LinearTimeSelfAttention(
                    dim_out,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys[6],
                )
            )
        else:
            self.attn = None

    def __call__(self, x: Array, t: Array, c: Optional[Array], *, key: Key) -> Array:
        C, _, _ = x.shape
        # In DDPM, each set of resblocks ends with an up/down sampling. In
        # biggan there is a final resblock after the up/downsampling. In this
        # code, the biggan approach is taken for both.
        # norm -> nonlinearity -> up/downsample -> conv follows Yang
        # https://github.dev/yang-song/score_sde/blob/main/models/layerspp.py
        h = jax.nn.silu(self.block1_groupnorm(x))
        if self.up or self.down:
            h = self.scaling(h)  # pyright: ignore
            x = self.scaling(x)  # pyright: ignore
        h = self.block1_conv(h)

        for layer in self.t_mlp_layers:
            t = layer(t)
        h += t[..., None, None]
        if c is not None:
            for layer in self.c_mlp_layers:
                c = layer(c)
            h += c[..., None, None]

        for layer in self.block2_layers:
            # precisely 1 dropout layer in block2_layers which requires a key
            if isinstance(layer, eqx.nn.Dropout):
                if key is None:
                    h = layer(h, inference=True)
                else:
                    h = layer(h, key=key)
            else:
                h = layer(h)

        if C != self.dim_out or self.up or self.down:
            x = self.res_conv(x)

        out = (h + x) / jnp.sqrt(2)
        if self.attn is not None:
            out = self.attn(out)
        return out


class UNet(AbstractNetwork):
    temb: GaussianFourierProjection
    t_mlp: eqx.nn.MLP
    c_mlp: eqx.nn.MLP | None
    first_conv: eqx.nn.Conv2d
    down_res_blocks: List[List[ResnetBlock]]
    mid_block1: ResnetBlock
    mid_block2: ResnetBlock
    ups_res_blocks: List[List[ResnetBlock]]
    final_conv_layers: List[Union[Callable, eqx.nn.LayerNorm, eqx.nn.Conv2d]]

    def __init__(
        self,
        data_shape: Tuple[int, int, int],
        is_biggan: bool,
        dim_mults: List[int],
        hidden_size: int,
        heads: int,
        dim_head: int,
        dropout_rate: float,
        num_res_blocks: int,
        attn_resolutions: List[int],
        *,
        key: Key,
        is_conditional: bool = True,
    ):
        keys = jax.random.split(key, 9)
        del key

        data_channels, in_height, in_width = data_shape

        dims = [hidden_size] + [hidden_size * m for m in dim_mults]
        in_out = list(exact_zip(dims[:-1], dims[1:]))

        # setup time handling
        self.temb = GaussianFourierProjection(hidden_size, key=keys[0])
        self.t_mlp = eqx.nn.MLP(
            hidden_size,
            hidden_size,
            4 * hidden_size,
            1,
            activation=jax.nn.silu,
            key=keys[1],
        )

        # setup conditional handling (no positional encoding here)
        if is_conditional:
            self.c_mlp = eqx.nn.MLP(
                hidden_size,
                hidden_size,
                4 * hidden_size,
                1,
                activation=jax.nn.silu,
                key=keys[2],
            )
        else:
            self.c_mlp = None

        # lifting layer
        self.first_conv = eqx.nn.Conv2d(
            data_channels, hidden_size, kernel_size=3, padding=1, key=keys[3]
        )

        h, w = in_height, in_width

        # setup resnet blocks for downsampling
        self.down_res_blocks = []
        num_keys = len(in_out) * num_res_blocks - 1
        keys_resblock = jr.split(keys[4], num_keys)
        i = 0
        for ind, (dim_in, dim_out) in enumerate(in_out):
            if h in attn_resolutions and w in attn_resolutions:
                is_attn = True
            else:
                is_attn = False
            res_blocks = [
                ResnetBlock(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    is_biggan=is_biggan,
                    up=False,
                    down=False,
                    time_emb_dim=hidden_size,
                    cond_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    is_attn=is_attn,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys_resblock[i],
                    is_conditional=is_conditional,
                )
            ]
            i += 1
            for _ in range(num_res_blocks - 2):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_out,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=False,
                        time_emb_dim=hidden_size,
                        cond_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                        is_conditional=is_conditional,
                    )
                )
                i += 1
            if ind < (len(in_out) - 1):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_out,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=True,
                        time_emb_dim=hidden_size,
                        cond_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                        is_conditional=is_conditional,
                    )
                )
                i += 1
                h, w = h // 2, w // 2
            self.down_res_blocks.append(res_blocks)
        assert i == num_keys

        # setup mid resnet blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            dim_in=mid_dim,
            dim_out=mid_dim,
            is_biggan=is_biggan,
            up=False,
            down=False,
            time_emb_dim=hidden_size,
            cond_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            is_attn=True,
            heads=heads,
            dim_head=dim_head,
            key=keys[5],
            is_conditional=is_conditional,
        )
        self.mid_block2 = ResnetBlock(
            dim_in=mid_dim,
            dim_out=mid_dim,
            is_biggan=is_biggan,
            up=False,
            down=False,
            time_emb_dim=hidden_size,
            cond_emb_dim=hidden_size,
            dropout_rate=dropout_rate,
            is_attn=False,
            heads=heads,
            dim_head=dim_head,
            key=keys[6],
            is_conditional=is_conditional,
        )

        # setup resnet blocks for upsampling
        self.ups_res_blocks = []
        num_keys = len(in_out) * (num_res_blocks + 1) - 1
        keys_resblock = jr.split(keys[7], num_keys)
        i = 0
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            if h in attn_resolutions and w in attn_resolutions:
                is_attn = True
            else:
                is_attn = False
            res_blocks = []
            for _ in range(num_res_blocks - 1):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_out * 2,
                        dim_out=dim_out,
                        is_biggan=is_biggan,
                        up=False,
                        down=False,
                        time_emb_dim=hidden_size,
                        cond_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                        is_conditional=is_conditional,
                    )
                )
                i += 1
            res_blocks.append(
                ResnetBlock(
                    dim_in=dim_out + dim_in,
                    dim_out=dim_in,
                    is_biggan=is_biggan,
                    up=False,
                    down=False,
                    time_emb_dim=hidden_size,
                    cond_emb_dim=hidden_size,
                    dropout_rate=dropout_rate,
                    is_attn=is_attn,
                    heads=heads,
                    dim_head=dim_head,
                    key=keys_resblock[i],
                    is_conditional=is_conditional,
                )
            )
            i += 1
            if ind < (len(in_out) - 1):
                res_blocks.append(
                    ResnetBlock(
                        dim_in=dim_in,
                        dim_out=dim_in,
                        is_biggan=is_biggan,
                        up=True,
                        down=False,
                        time_emb_dim=hidden_size,
                        cond_emb_dim=hidden_size,
                        dropout_rate=dropout_rate,
                        is_attn=is_attn,
                        heads=heads,
                        dim_head=dim_head,
                        key=keys_resblock[i],
                        is_conditional=is_conditional,
                    )
                )
                i += 1
                h, w = h * 2, w * 2

            self.ups_res_blocks.append(res_blocks)
        assert i == num_keys

        # final conv layers
        self.final_conv_layers = [
            eqx.nn.GroupNorm(min(hidden_size // 4, 32), hidden_size),
            jax.nn.silu,
            eqx.nn.Conv2d(hidden_size, data_channels, 1, key=keys[8]),
        ]

    def __call__(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        t = self.temb(t)
        t = self.t_mlp(t)
        if c is not None:
            assert self.c_mlp is not None
            c = self.temb(c)
            c = self.c_mlp(c)
        h = self.first_conv(x)
        hs = [h]
        for res_blocks in self.down_res_blocks:
            for res_block in res_blocks:
                key, subkey = key_split_allowing_none(key)
                h = res_block(h, t, c, key=subkey)
                hs.append(h)

        key, subkey = key_split_allowing_none(key)
        h = self.mid_block1(h, t, c, key=subkey)
        key, subkey = key_split_allowing_none(key)
        h = self.mid_block2(h, t, c, key=subkey)

        for res_blocks in self.ups_res_blocks:
            for res_block in res_blocks:
                key, subkey = key_split_allowing_none(key)
                if res_block.up:
                    h = res_block(h, t, c, key=subkey)
                else:
                    h = res_block(
                        jnp.concatenate((h, hs.pop()), axis=0), t, c, key=subkey
                    )

        assert len(hs) == 0

        for layer in self.final_conv_layers:
            h = layer(h)

        return h
