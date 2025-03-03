import functools as ft
from abc import ABC, abstractmethod
from typing import Tuple

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import AbstractSolver
from equinox import filter_jit
from jaxtyping import Array, Key

from .diffusion import AbstractDiffusionModel


class AbstractSampler(ABC):
    @abstractmethod
    def single_sample(
        self,
        model: AbstractDiffusionModel,
        data_shape: Tuple[int, ...],
        conds: Array | None,
        key: Key,
    ) -> Array:
        pass

    # limamau: add possibility to use modifiers and use explicitly
    # the score instead of getting it from the model
    def sample(
        self,
        model: AbstractDiffusionModel,
        data_shape: Tuple[int, ...],
        conds: Array | None,
        key: Key,
        norm_mean: Array,
        norm_std: Array,
        sample_size: int,
    ) -> Array:
        sample_key = jr.split(key, sample_size)
        sample_fn = ft.partial(
            self.single_sample,
            model,
            data_shape,
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
        conds: Array | None,
        key: Key,
    ) -> Array:
        def fn(t, x, args):
            f = model.drift(x, t)
            g2 = jnp.square(model.diffusion(t))
            s = model.score(x, t, conds)
            return f - 0.5 * g2 * s

        term = dfx.ODETerm(fn)
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
