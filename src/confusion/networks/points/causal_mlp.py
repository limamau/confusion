from typing import List, Optional

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, Key, Real

from ..layers import GaussianFourierProjection
from ..networks import AbstractCausalNetwork


class VariablePartitionedLinear(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, in_dim: int, out_dim: int, key: Key):
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)

    def __call__(
        self, x: Float[Array, "  vars_dim in_dim"]
    ) -> Float[Array, " vars_dim out_dim"]:
        x = jax.vmap(self.linear)(x)
        return x


class CausalAttention(eqx.Module):
    causal_mask: Bool[Array, " vars_dim vars_dim"]
    in_dim: int
    qkv_dim: int
    attention: eqx.nn.MultiheadAttention
    in_linear: VariablePartitionedLinear
    out_linear: VariablePartitionedLinear

    def __init__(
        self,
        causal_mask: Bool[Array, " vars_dim vars_dim"],
        in_dim: int,
        qkv_dim: int,
        key: Key,
    ):
        keys = jr.split(key, 3)
        self.causal_mask = causal_mask
        self.in_dim = in_dim
        self.qkv_dim = qkv_dim
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=1,
            query_size=qkv_dim,
            key=keys[0],
        )
        self.in_linear = VariablePartitionedLinear(in_dim, qkv_dim * 3, key=keys[1])
        self.out_linear = VariablePartitionedLinear(qkv_dim, in_dim, key=keys[2])

    def __call__(
        self, x: Float[Array, " vars_dim in_dim"]
    ) -> Float[Array, " vars_dim in_dim"]:
        qkv = self.in_linear(x)  # [(vars_dim, qkv_dim*3,)]
        q, k, v = jnp.split(qkv, 3, axis=1)  # ((vars_dim, qkv_dim,), ...)
        x = self.attention(q, k, v, mask=self.causal_mask)  # (vars_dim, qkv_dim)
        x = self.out_linear(x)  # [(vars_dim, in_dim)]
        return x


class CausalMultiLayerPerceptron(AbstractCausalNetwork):
    vars_dim: int
    num_blocks: int
    temb: GaussianFourierProjection
    in_linear: VariablePartitionedLinear
    hidden_linears: List[VariablePartitionedLinear]
    causal_attentions: List[CausalAttention]
    out_linear1: VariablePartitionedLinear
    out_linear2: VariablePartitionedLinear

    def __init__(
        self,
        num_blocks: int,
        vars_dim: int,
        hidden_dim: int,
        temb_dim: int,
        projection_scale: float,
        causal_mask: Bool[Array, "..."],
        num_heads: int,
        qkv_size: int,
        *,
        key: Key,
        is_conditional: bool = False,
    ):
        # ints
        self.vars_dim = vars_dim
        self.num_blocks = num_blocks

        # keys for initialization
        keys = jr.split(key, 4 + 2 * num_blocks)

        # pre-conditioning or not
        if is_conditional:
            in_channels = temb_dim + 2
        else:
            in_channels = temb_dim + 1

        # time
        self.temb = GaussianFourierProjection(temb_dim, projection_scale, key=keys[0])
        self.in_linear = VariablePartitionedLinear(in_channels, hidden_dim, key=keys[1])

        # hidden linears
        self.hidden_linears = []
        for b in range(num_blocks):
            self.hidden_linears.append(
                VariablePartitionedLinear(hidden_dim, hidden_dim, key=keys[2 + 2 * b])
            )

        # causal attention
        self.causal_attentions = []
        for b in range(num_blocks):
            self.causal_attentions.append(
                CausalAttention(causal_mask, hidden_dim, qkv_size, key=keys[3 + 2 * b])
            )

        # aggregation
        self.out_linear1 = VariablePartitionedLinear(
            hidden_dim * num_blocks, hidden_dim, key=keys[2 + 2 * num_blocks]
        )
        self.out_linear2 = VariablePartitionedLinear(
            hidden_dim, 1, key=keys[3 + 2 * num_blocks]
        )

    def __call__(
        self,
        x: Float[Array, " vars_dim"],
        t: Float[Array, " "],
        c: Optional[Real[Array, " conds_dim"]],
        *,
        key: Optional[Key] = None,
    ) -> Float[Array, " vars_dim"]:
        # time embedding
        t = self.temb(t)

        # expand dims in order to use vmaps on vars dim
        x = jnp.expand_dims(x, axis=1)

        # aggregate time and conditions to each variable
        t = einops.repeat(t, " temb_dim -> vars_dim temb_dim", vars_dim=self.vars_dim)
        if c is None:
            x = jnp.concatenate([x, t], axis=-1)
        else:
            c = einops.repeat(
                c, " conds_dim -> vars_dim conds_dim", vars_dim=self.vars_dim
            )
            x = jnp.concatenate([x, t, c], axis=-1)

        # lifting layer
        x = self.in_linear(x)

        # loop over main blocks
        att_list = []
        for b in range(self.num_blocks):
            x = self.hidden_linears[b](x)
            x = self.causal_attentions[b](x)
            att_list.append(x)

        # aggregation
        x = jnp.concatenate(att_list, axis=-1)
        x = self.out_linear1(x)
        x = self.out_linear2(x)
        x = jnp.squeeze(x, axis=-1)
        return x
