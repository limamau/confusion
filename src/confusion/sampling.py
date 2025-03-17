import functools as ft
from abc import abstractmethod
from typing import Optional, Tuple

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import AbstractSolver, AbstractStepSizeController
from equinox import filter_jit
from jaxtyping import Array, Key

from .diffusion import AbstractDiffusionModel
from .guidance import AbstractGuidance, GuidanceFree


# samplers #
class AbstractSampler:
    dt0: float
    t0: float
    t1: float
    solver: AbstractSolver
    stepsize_controller: AbstractStepSizeController

    def __init__(
        self,
        dt0: float,
        solver: AbstractSolver,
        t0: float = 1e-3,
        t1: float = 1.0,
        stepsize_controller: AbstractStepSizeController = dfx.ConstantStepSize(),
    ):
        self.dt0 = dt0
        self.solver = solver
        self.t0 = t0
        self.t1 = t1
        self.stepsize_controller = stepsize_controller

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
    def __init__(
        self,
        dt0: float,
        solver: AbstractSolver = dfx.Tsit5(),
        t0: float = 1e-3,
        t1: float = 1.0,
        step_size_controller: AbstractStepSizeController = dfx.ConstantStepSize(),
    ):
        super().__init__(dt0, solver, t0, t1, step_size_controller)

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
        x1 = jr.normal(key, data_shape)
        # solve from t1 to t0
        sol = dfx.diffeqsolve(
            term,
            self.solver,
            self.t1,
            self.t0,
            -self.dt0,
            x1,
            stepsize_controller=self.stepsize_controller,
        )

        assert sol.ys is not None
        return sol.ys[0]


class SDESampler(AbstractSampler):
    def __init__(
        self,
        dt0: float,
        solver: AbstractSolver = dfx.Euler(),
        t0: float = 1e-3,
        t1: float = 1.0,
        step_size_controller: AbstractStepSizeController = dfx.ConstantStepSize(),
    ):
        super().__init__(dt0, solver, t0, t1, step_size_controller)

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
        bm = dfx.VirtualBrownianTree(self.t0, 1.0, tol=self.dt0, shape=(), key=keys[0])
        terms = dfx.MultiTerm(
            dfx.ODETerm(back_drift), dfx.ControlTerm(back_diffusion, bm)
        )
        x1 = jr.normal(keys[1], data_shape)
        # solve from t1=1.0 to t0=t0
        sol = dfx.diffeqsolve(
            terms,
            solver,
            self.t1,
            self.t0,
            -self.dt0,
            x1,
            stepsize_controller=self.stepsize_controller,
        )

        assert sol.ys is not None
        return sol.ys[0]
