from typing import Tuple

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from diffrax import AbstractSolver
from jaxtyping import Array, Key

from .diffusion import AbstractDiffusionModel

# TODO: code this
# class AbstractSampler(ABC):
#     @abstractmethod
#     @staticmethod
#     def single_sample_fn(
#         model: AbstractDiffusionModel,
#         data_shape: Tuple[int, ...],
#         dt0: float,
#         t1: float,
#         conds: Array | None,
#         key: Key,
#     ) -> Array:
#         pass

#     def sample(
#         self,
#         model: AbstractDiffusionModel,
#         data_shape: Tuple[int, ...],
#         dt0: float,
#         t1: float,
#         conds: Array | None,
#         key: Key,
#         norm_mean: Array,
#         norm_std: Array,
#         sample_size: int,
#     ) -> Array:
#         sample_key = jr.split(key, sample_size)
#         sample_fn = partial(
#             self.single_sample_fn,
#             model,
#             norm_mean.shape,
#             dt0,
#             t1,
#         )
#         gen_samples = jax.vmap(sample_fn)(conds, sample_key)
#         return norm_mean + norm_std * gen_samples


@eqx.filter_jit
def single_ode_sample_fn(
    model: AbstractDiffusionModel,
    data_shape: Tuple[int, ...],
    dt0: float,
    t1: float,
    conds: Array | None,
    key: Key,
    solver: AbstractSolver = dfx.Tsit5(),
) -> Array:
    def fn(t, x, args):
        f = model.drift(x, t)
        g2 = jnp.square(model.diffusion(t))
        s = model.score(x, t, conds)
        return f - 0.5 * g2 * s

    term = dfx.ODETerm(fn)
    t0 = 0
    x1 = jr.normal(key, data_shape)
    # solve from t1 to t0
    sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, x1)

    assert sol.ys is not None
    return sol.ys[0]


@eqx.filter_jit
def single_sde_sample_fn(
    model: AbstractDiffusionModel,
    data_shape: Tuple[int, ...],
    dt0: float,
    t1: float,
    conds: Array | None,
    key: Key,
    solver: AbstractSolver = dfx.Euler(),
) -> Array:
    def back_drift(t, x, args):
        f = model.drift(x, t)
        g2 = jnp.square(model.diffusion(t))
        s = model.score(x, t, conds)
        return f - g2 * s

    def back_diffusion(t, x, args):
        return model.diffusion(t)

    keys = jr.split(key, 2)
    t0 = 0
    bm = dfx.VirtualBrownianTree(t0, t1, tol=dt0, shape=(), key=keys[0])
    terms = dfx.MultiTerm(dfx.ODETerm(back_drift), dfx.ControlTerm(back_diffusion, bm))
    x1 = jr.normal(keys[1], data_shape)
    # solve from t1 to t0
    sol = dfx.diffeqsolve(terms, solver, t1, t0, -dt0, x1)

    assert sol.ys is not None
    return sol.ys[0]
