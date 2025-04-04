import functools as ft
from abc import abstractmethod
from typing import Optional, Tuple

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from diffrax import AbstractSolver, AbstractStepSizeController
from equinox import filter_jit
from jaxtyping import Array, Key

from .diffusion import AbstractDiffusionModel
from .guidance import AbstractGuidance, GuidanceFree
from .sdes import AbstractSDE
from .utils import denormalize


def edm_sampling_ts(
    sde: AbstractSDE,
    rho: float = 7.0,
    N: int = 300,
    t0: float = 1e-3,
    t1: float = 1.0,
):
    sigma_max = sde.sigma(jnp.array([t1]))
    sigma_min = sde.sigma(jnp.array([t0]))

    sigmas = jnp.power(
        jnp.power(sigma_max, (1 / rho))
        + jnp.arange(N)
        / (N - 1)
        * (jnp.power(sigma_min, (1 / rho)) - jnp.power(sigma_max, (1 / rho))),
        rho,
    )

    ts = jnp.array([sde.t(sigma_i) for sigma_i in sigmas])

    # to bypass dfx.diffeqsolve's assertion
    if abs(ts[0] - t1) < 1e-2 and abs(ts[-1] - t0) < 1e-4:
        ts = ts.at[0].set(t1)
        ts = ts.at[-1].set(t0)
    else:
        raise ValueError(
            "ts must start at t1 and end at t0, but got [{}, {}].".format(ts[0], ts[-1])
        )

    return ts


# samplers #
class AbstractSampler:
    dt0: Optional[float]
    t0: float
    t1: float

    def __init__(
        self,
        dt0: Optional[float],
        t0: float = 1e-3,
        t1: float = 1.0,
        *args,
        **kwargs,
    ):
        self.dt0 = dt0
        self.t0 = t0
        self.t1 = t1

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
        num_samples: int,
        guidance: AbstractGuidance = GuidanceFree(),
    ) -> Array:
        sample_key = jr.split(key, num_samples)
        sample_fn = ft.partial(
            self.single_sample,
            model,
            data_shape,
            guidance,
        )
        gen_samples = jax.vmap(sample_fn)(conds, sample_key)
        return denormalize(gen_samples, norm_mean, norm_std, model.sigma_data)


class ODESampler(AbstractSampler):
    solver: AbstractSolver
    stepsize_controller: AbstractStepSizeController

    def __init__(
        self,
        dt0: Optional[float],
        t0: float = 1e-3,
        t1: float = 1.0,
        solver: AbstractSolver = dfx.Tsit5(),
        stepsize_controller: AbstractStepSizeController = dfx.ConstantStepSize(),
    ):
        super().__init__(dt0, t0, t1)
        self.solver = solver
        self.stepsize_controller = stepsize_controller

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
            f = model.sde.drift(x, t)
            g2 = jnp.square(model.sde.diffusion(t))
            score = guidance.apply(model, x, t, conds, key=None)
            return f - 0.5 * g2 * score

        term = dfx.ODETerm(fun)

        # solve from t1 to t0
        x1 = jr.normal(key, data_shape)
        dt0 = -self.dt0 if self.dt0 is not None else None
        sol = dfx.diffeqsolve(
            term,
            self.solver,
            self.t1,
            self.t0,
            dt0,
            x1,
            stepsize_controller=self.stepsize_controller,
        )

        assert sol.ys is not None
        return sol.ys[0]


class SDESampler(AbstractSampler):
    solver: AbstractSolver
    stepsize_controller: AbstractStepSizeController

    def __init__(
        self,
        dt0: Optional[float],
        t0: float = 1e-3,
        t1: float = 1.0,
        solver: AbstractSolver = dfx.Euler(),
        stepsize_controller: AbstractStepSizeController = dfx.ConstantStepSize(),
    ):
        super().__init__(dt0, t0, t1)
        self.solver = solver
        self.stepsize_controller = stepsize_controller

    @filter_jit
    def single_sample(
        self,
        model: AbstractDiffusionModel,
        data_shape: Tuple[int, ...],
        guidance: AbstractGuidance,
        conds: Optional[Array],
        key: Key,
    ) -> Array:
        def back_drift(t, x, args):
            f = model.sde.drift(x, t)
            g2 = jnp.square(model.sde.diffusion(t))
            score = guidance.apply(model, x, t, conds, key=key)
            return f - g2 * score

        def back_diffusion(t, x, args):
            return model.sde.diffusion(t)

        keys = jr.split(key, 2)
        bm = dfx.VirtualBrownianTree(self.t0, 1.0, tol=1e-3, shape=(), key=keys[0])
        terms = dfx.MultiTerm(
            dfx.ODETerm(back_drift), dfx.ControlTerm(back_diffusion, bm)
        )

        # solve from t1 to t0
        x1 = jr.normal(keys[1], data_shape)
        back_dt0 = -self.dt0 if self.dt0 is not None else None
        sol = dfx.diffeqsolve(
            terms,
            self.solver,
            self.t1,
            self.t0,
            back_dt0,
            x1,
            stepsize_controller=self.stepsize_controller,
        )

        assert sol.ys is not None
        return sol.ys[0]


# the class below is basically a home-made implementation of the Euler-Maruyama solver,
# which can be equally achived with the SDESampler class above using dfx.Euler()
class EulerMaruyamaSampler(AbstractSampler):
    def __init__(
        self,
        dt0: Optional[float] = 5e-3,
        t0: float = 1e-3,
        t1: float = 1.0,
    ):
        super().__init__(dt0, t0, t1)

    @filter_jit
    def single_sample(
        self,
        model: AbstractDiffusionModel,
        data_shape: Tuple[int, ...],
        guidance: AbstractGuidance,
        conds: Optional[Array],
        key: Key,
    ) -> Array:
        if self.dt0 is None:
            raise ValueError("dt0 must be provided for Euler–Maruyama sampling.")

        def back_drift(t, x):
            f = model.sde.drift(x, t)
            g2 = jnp.square(model.sde.diffusion(t))
            score = guidance.apply(model, x, t, conds, key=key)
            return f - g2 * score

        def back_diffusion(t, x):
            return model.sde.diffusion(t)

        # pre-allocations
        x1 = jr.normal(key, data_shape)
        total_time = self.t1 - self.t0
        num_steps = np.ceil(total_time / self.dt0).astype(int)
        dt = total_time / num_steps
        t_array = jnp.linspace(self.t1, self.t0, num=num_steps + 1)
        keys = jr.split(key, num_steps + 1)

        # solve from t1 to t0
        def euler_step(x, i):
            t_i = t_array[i]
            eta = jr.normal(keys[i], data_shape)
            drift_val = back_drift(t_i, x)
            diff_val = back_diffusion(t_i, x)
            x_next = x + drift_val * (-dt) + diff_val * jnp.sqrt(dt) * eta
            return x_next, ()  # (new_carry, output)

        x, _ = jax.lax.scan(euler_step, x1, jnp.arange(num_steps))

        return x
