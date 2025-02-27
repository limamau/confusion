from typing import Callable, Tuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import AbstractSolver
from jaxtyping import Array, Key


@eqx.filter_jit
def single_ode_sample_fn(
    model: eqx.Module,
    int_beta: Callable,
    data_shape: Tuple[int, ...],
    dt0: float,
    t1: float,
    conds: Array | None,
    key: Key,
    solver: AbstractSolver = dfx.Tsit5(),
) -> Array:
    def drift(t, y, args):
        # get beta by derivating the integral
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * (y + model(y, t, conds))

    term = dfx.ODETerm(drift)
    t0 = 0
    y1 = jr.normal(key, data_shape)
    # solve from t1 to t0
    sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, y1)

    assert sol.ys is not None
    return sol.ys[0]


@eqx.filter_jit
def single_sde_sample_fn(
    model: eqx.Module,
    int_beta: Callable,
    data_shape: Tuple[int, ...],
    dt0: float,
    t1: float,
    conds: Array | None,
    key: Key,
    solver: AbstractSolver = dfx.Euler(),
) -> Array:
    def drift(t, y, args):
        # get beta by derivating the integral
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        return -beta * (0.5 * y + model(y, t, conds))

    def diffusion(t, y, args):
        # get beta by derivating the integral
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        return jnp.sqrt(beta) * jnp.ones_like(y)

    keys = jr.split(key, 2)
    t0 = 0
    bm = dfx.VirtualBrownianTree(t0, t1, tol=dt0, shape=(), key=keys[0])
    terms = dfx.MultiTerm(dfx.ODETerm(drift), dfx.ControlTerm(diffusion, bm))
    y1 = jr.normal(keys[1], data_shape)
    # solve from t1 to t0
    sol = dfx.diffeqsolve(terms, solver, t1, t0, -dt0, y1)

    assert sol.ys is not None
    return sol.ys[0]
