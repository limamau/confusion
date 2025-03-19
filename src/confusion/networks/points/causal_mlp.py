from typing import List, Optional, Union

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, Key, Real

from ..layers import GaussianFourierProjection
from ..networks import AbstractNetwork


class PerVariableSharedLinear(eqx.Module):
    linear: eqx.nn.Linear

    # here vars_size is not used, but passed as a parameter for consistency
    def __init__(self, vars_size: int, in_size: int, out_size: int, *, key: Key):
        self.linear = eqx.nn.Linear(in_size, out_size, key=key)

    def __call__(
        self, x: Float[Array, " vars_size in_size"]
    ) -> Float[Array, " vars_size out_size"]:
        x = jax.vmap(self.linear)(x)
        return x


class PerVariableIndependentLinears(eqx.Module):
    linears: list[eqx.nn.Linear]
    num_vars: int

    def __init__(self, vars_size: int, in_size: int, out_size: int, *, key: Key):
        keys = jax.random.split(key, vars_size)
        self.linears = [
            eqx.nn.Linear(in_size, out_size, key=keys[i]) for i in range(vars_size)
        ]
        self.num_vars = vars_size

    def __call__(
        self, x: Float[Array, " vars_size in_size"]
    ) -> Float[Array, " vars_size out_size"]:
        results = [self.linears[i](x[i]) for i in range(self.num_vars)]
        return jnp.stack(results)


# common type for the two above classes
PerVariableLinear = Union[PerVariableSharedLinear, PerVariableIndependentLinears]


class CausalAttention(eqx.Module):
    causal_mask: Bool[Array, " vars_size vars_size"]
    in_size: int
    qkv_size: int
    attention: eqx.nn.MultiheadAttention
    in_linear: PerVariableLinear
    out_linear: PerVariableLinear

    def __init__(
        self,
        linear_class: type[PerVariableSharedLinear]
        | type[PerVariableIndependentLinears],
        vars_size: int,
        causal_mask: Bool[Array, " vars_size vars_size"],
        in_size: int,
        qkv_size: int,
        key: Key,
    ):
        keys = jr.split(key, 3)
        self.causal_mask = causal_mask
        self.in_size = in_size
        self.qkv_size = qkv_size
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=1,
            query_size=qkv_size,
            key=keys[0],
        )
        self.in_linear = linear_class(vars_size, in_size, qkv_size * 3, key=keys[1])
        self.out_linear = linear_class(vars_size, qkv_size, in_size, key=keys[2])

    def __call__(
        self, x: Float[Array, " vars_size in_size"]
    ) -> Float[Array, " vars_size in_size"]:
        qkv = self.in_linear(x)  # [(vars_size, qkv_size*3,)]
        q, k, v = jnp.split(qkv, 3, axis=1)  # ((vars_size, qkv_size,), ...)
        x = self.attention(q, k, v, mask=self.causal_mask)  # (vars_size, qkv_size)
        x = self.out_linear(x)  # [(vars_size, in_size)]
        return x


class CausalMultiLayerPerceptron(AbstractNetwork):
    vars_size: int
    num_blocks: int
    temb: GaussianFourierProjection
    in_linear: PerVariableLinear
    hidden_linears: List[PerVariableLinear]
    causal_attentions: List[CausalAttention]
    out_linear1: PerVariableLinear
    out_linear2: PerVariableLinear

    def __init__(
        self,
        num_blocks: int,
        vars_size: int,
        hidden_size: int,
        temb_size: int,
        projection_scale: float,
        causal_mask: Bool[Array, "..."],
        num_heads: int,
        qkv_size: int,
        *,
        key: Key,
        is_conditional: bool = False,
        use_shared_linears: bool = False,
    ):
        # ints
        self.vars_size = vars_size
        self.num_blocks = num_blocks

        # keys for initialization
        keys = jr.split(key, 4 + 2 * num_blocks)

        # pre-conditioning or not
        if is_conditional:
            in_channels = temb_size + 2
        else:
            in_channels = temb_size + 1

        # choose linear type
        linear_class = (
            PerVariableSharedLinear
            if use_shared_linears
            else PerVariableIndependentLinears
        )

        # time
        self.temb = GaussianFourierProjection(temb_size, projection_scale, key=keys[0])
        self.in_linear = linear_class(vars_size, in_channels, hidden_size, key=keys[1])

        # hidden linears
        self.hidden_linears = []
        for b in range(num_blocks):
            self.hidden_linears.append(
                linear_class(vars_size, hidden_size, hidden_size, key=keys[2 + 2 * b])
            )

        # causal attention
        self.causal_attentions = []
        for b in range(num_blocks):
            self.causal_attentions.append(
                CausalAttention(
                    linear_class,
                    vars_size,
                    causal_mask,
                    hidden_size,
                    qkv_size,
                    key=keys[3 + 2 * b],
                )
            )

        # aggregation
        self.out_linear1 = linear_class(
            vars_size,
            hidden_size * num_blocks,
            hidden_size,
            key=keys[2 + 2 * num_blocks],
        )
        self.out_linear2 = linear_class(
            vars_size, hidden_size, 1, key=keys[3 + 2 * num_blocks]
        )

    def __call__(
        self,
        x: Float[Array, " vars_size"],
        t: Float[Array, " "],
        c: Optional[Real[Array, " conds_size"]],
        *,
        key: Optional[Key] = None,
    ) -> Float[Array, " vars_size"]:
        # time embedding
        t = self.temb(t)

        # expand dims in order to use vmaps on vars dim
        x = jnp.expand_dims(x, axis=1)

        # aggregate time and conditions to each variable
        t = einops.repeat(
            t, " temb_size -> vars_size temb_size", vars_size=self.vars_size
        )
        if c is None:
            x = jnp.concatenate([x, t], axis=-1)
        else:
            c = einops.repeat(
                c, " conds_size -> vars_size conds_size", vars_size=self.vars_size
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
