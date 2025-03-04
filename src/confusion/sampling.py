import functools as ft
from abc import abstractmethod
from typing import Optional, Tuple

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import AbstractSolver
from equinox import filter_jit
from jaxtyping import Array, Key

from .diffusion import AbstractDiffusionModel
from .guidance import AbstractGuidance, GuidanceFree


class AbstractSampler:
    @abstractmethod
    def single_sample(
        self,
        model: AbstractDiffusionModel,
        data_shape: Tuple[int, ...],
        guidance: AbstractGuidance,
        conds: Optional[Array],
        key: Key,
    ) -> Array:
        raise NotImplementedError

    def sample(
        self,
        model: AbstractDiffusionModel,
        data_shape: Tuple[int, ...],
        conds: Optional[Array],
        key: Key,
        norm_mean: Array,
        norm_std: Array,
        sample_size: int,
        guidance: AbstractGuidance = GuidanceFree(),
    ) -> Array:
        sample_key = jr.split(key, sample_size)
        sample_fn = ft.partial(
            self.single_sample,
            model,
            data_shape,
            guidance,
        )
        gen_samples = jax.vmap(sample_fn)(conds, sample_key)  # pyright: ignore
        return norm_mean + norm_std * gen_samples


class ODESampler(AbstractSampler):
    dt0: float
    t1: float
    solver: AbstractSolver

    def __init__(self, dt0: float, t1: float, solver: AbstractSolver = dfx.Tsit5()):
        self.dt0 = dt0
        self.t1 = t1
        self.solver = solver

    @filter_jit
    def single_sample(
        self,
        model: AbstractDiffusionModel,
        data_shape: Tuple[int, ...],
        guidance: AbstractGuidance,
        conds: Optional[Array],
        key: Key,
    ) -> Array:
        def fun(t, x, args):
            f = model.drift(x, t)
            g2 = jnp.square(model.diffusion(t))
            score = guidance.apply(model, x, t, conds, key=None)
            return f - 0.5 * g2 * score

        term = dfx.ODETerm(fun)
        t0 = model.t0
        x1 = jr.normal(key, data_shape)
        # solve from t1 to t0
        sol = dfx.diffeqsolve(term, self.solver, self.t1, t0, -self.dt0, x1)

        assert sol.ys is not None
        return sol.ys[0]


class SDESampler(AbstractSampler):
    dt0: float
    t1: float
    solver: AbstractSolver

    def __init__(self, dt0: float, t1: float, solver: AbstractSolver = dfx.Euler()):
        self.dt0 = dt0
        self.t1 = t1
        self.solver = solver

    @filter_jit
    def single_sample(
        self,
        model: AbstractDiffusionModel,
        data_shape: Tuple[int, ...],
        guidance: AbstractGuidance,
        conds: Optional[Array],
        key: Key,
        solver: AbstractSolver = dfx.Euler(),
    ) -> Array:
        def back_drift(t, x, args):
            f = model.drift(x, t)
            g2 = jnp.square(model.diffusion(t))
            score = guidance.apply(model, x, t, conds, key=key)
            return f - g2 * score

        def back_diffusion(t, x, args):
            return model.diffusion(t)

        keys = jr.split(key, 2)
        t0 = model.t0
        bm = dfx.VirtualBrownianTree(t0, self.t1, tol=self.dt0, shape=(), key=keys[0])
        terms = dfx.MultiTerm(
            dfx.ODETerm(back_drift), dfx.ControlTerm(back_diffusion, bm)
        )
        x1 = jr.normal(keys[1], data_shape)
        # solve from t1 to t0
        sol = dfx.diffeqsolve(terms, solver, self.t1, t0, -self.dt0, x1)

        assert sol.ys is not None
        return sol.ys[0]
