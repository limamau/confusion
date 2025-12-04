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

from confusion.guidance import AbstractGuidance, GuidanceFree
from confusion.models.abstract import AbstractModel


class AbstractSampler:
    @abstractmethod
    def single_sample(
        self,
        model: AbstractModel,
        data_shape: Tuple[int, ...],
        guidance: AbstractGuidance,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        key: Key,
    ) -> Array:
        raise NotImplementedError

    def sample(
        self,
        model: AbstractModel,
        data_shape: Tuple[int, ...],
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        key: Key,
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
        return jax.vmap(sample_fn)(pre_conds, post_conds, sample_key)


class AbstractODESampler(AbstractSampler):
    pass


class AbstractSDESampler(AbstractSampler):
    pass


class ConstantStepEulerMaruyamaSampler(AbstractSDESampler):
    def __init__(
        self,
        dt0: float = 1e-3,
        t0: float = 1e-3,
        t1: float = 1.0,
    ):
        self.dt0 = dt0
        self.t0 = t0
        self.t1 = t1

    @filter_jit
    def single_sample(
        self,
        model: AbstractModel,
        data_shape: Tuple[int, ...],
        guidance: AbstractGuidance,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        key: Key,
    ) -> Array:
        # pre-allocations
        sigma_max = model.sigma_max
        x1 = jr.normal(key, data_shape) * sigma_max
        total_time = self.t1 - self.t0
        num_steps = np.ceil(total_time / self.dt0).astype(int)
        dt = total_time / num_steps
        t_array = jnp.linspace(self.t1, self.t0, num=num_steps + 1)
        keys = jr.split(key, num_steps + 1)

        # solve from t1 to t0
        def euler_step(x, i):
            t_i = t_array[i]
            eta = jr.normal(keys[i], data_shape)

            drift_val, diff_val = model.reverse_sde(
                x, t_i, guidance, pre_conds, post_conds
            )

            diffeq = model.diffeq

            x_next = x + drift_val * (-dt) + diff_val * jnp.sqrt(dt) * eta
            x_next = guidance.apply_on_x_next(
                diffeq, x_next, t_i, pre_conds, post_conds, key=None
            )
            return x_next, ()  # (new_carry, output)

        x, _ = jax.lax.scan(euler_step, x1, jnp.arange(num_steps))

        return x


class ScheduledEulerMaruyamaSampler(AbstractSDESampler):
    def __init__(self, times: Array):
        self.times = times

    @filter_jit
    def single_sample(
        self,
        model: AbstractModel,
        data_shape: Tuple[int, ...],
        guidance: AbstractGuidance,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        key: Key,
    ) -> Array:
        # pre-allocations
        sigma_max = model.sigma_max
        x1 = jr.normal(key, data_shape) * sigma_max
        t_array = self.times[::-1]
        num_steps = len(self.times) - 1
        keys = jr.split(key, num_steps)

        # solve from t1 to t0
        def euler_step(x, i):
            t_i = t_array[i]
            eta = jr.normal(keys[i], data_shape)

            drift_val, diff_val = model.reverse_sde(
                x, t_i, guidance, pre_conds, post_conds
            )

            diffeq = model.diffeq

            dt = t_array[i - 1] - t_i
            x_next = x + drift_val * (-dt) + diff_val * jnp.sqrt(dt) * eta
            x_next = guidance.apply_on_x_next(
                diffeq, x_next, t_i, pre_conds, post_conds, key=None
            )
            return x_next, ()  # (new_carry, output)

        x, _ = jax.lax.scan(euler_step, x1, jnp.arange(1, num_steps + 1))

        return x


class ScheduledEulerSampler(AbstractODESampler):
    def __init__(self, times: Array):
        self.times = times

    @filter_jit
    def single_sample(
        self,
        model: AbstractModel,
        data_shape: Tuple[int, ...],
        guidance: AbstractGuidance,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        key: Key,
    ) -> Array:
        # pre-allocations
        sigma_max = model.sigma_max
        x1 = jr.normal(key, data_shape) * sigma_max
        t_array = self.times[::-1]
        num_steps = len(self.times) - 1

        # solve from t1 to t0
        def euler_step(x, i):
            t_i = t_array[i]
            drift_val = model.probability_flow_ode(
                x, t_i, guidance, pre_conds, post_conds
            )

            diffeq = model.diffeq

            dt = t_array[i - 1] - t_i
            x_next = x + drift_val * (-dt)
            x_next = guidance.apply_on_x_next(
                diffeq, x_next, t_i, pre_conds, post_conds, key=None
            )
            return x_next, ()  # (new_carry, output)

        x, _ = jax.lax.scan(euler_step, x1, jnp.arange(1, num_steps + 1))

        return x


class DebuggingEulerMaruyamaSampler(AbstractSDESampler):
    def __init__(self, times: Array):
        self.times = times

    @filter_jit
    def single_sample(
        self,
        model: AbstractModel,
        data_shape: Tuple[int, ...],
        guidance: AbstractGuidance,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        key: Key,
    ) -> Array:
        # pre-allocations
        sigma_max = model.sigma_max
        x1 = jr.normal(key, data_shape) * sigma_max
        t_array = self.times[::-1]
        num_steps = len(self.times) - 1
        keys = jr.split(key, num_steps)
        all_steps = jnp.zeros((num_steps + 1,) + data_shape)
        all_steps = all_steps.at[0].set(x1)

        # solve from t1 to t0
        def euler_step(carry, i):
            x, all_x = carry
            t_i = t_array[i]
            eta = jr.normal(keys[i - 1], data_shape)

            drift_val, diff_val = model.reverse_sde(
                x, t_i, guidance, pre_conds, post_conds
            )

            diffeq = model.diffeq

            dt = t_array[i - 1] - t_i
            x_next = x + drift_val * (-dt) + diff_val * jnp.sqrt(dt) * eta
            x_next = guidance.apply_on_x_next(
                diffeq, x_next, t_i, pre_conds, post_conds, key=None
            )
            all_x = all_x.at[i].set(x_next)
            return (x_next, all_x), x_next  # ((new_x, all_x), output_for_scan)

        # initial carry contains both the current x and the array of all steps
        (_, all_steps), _ = jax.lax.scan(
            euler_step, (x1, all_steps), jnp.arange(1, num_steps + 1)
        )

        return all_steps


class DebuggingEulerSampler(AbstractODESampler):
    def __init__(self, times: Array):
        self.times = times

    @filter_jit
    def single_sample(
        self,
        model: AbstractModel,
        data_shape: Tuple[int, ...],
        guidance: AbstractGuidance,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        key: Key,
    ) -> Array:
        # pre-allocations
        sigma_max = model.sigma_max
        x1 = jr.normal(key, data_shape) * sigma_max
        t_array = self.times[::-1]
        num_steps = len(self.times) - 1
        all_steps = jnp.zeros((num_steps + 1,) + data_shape)
        all_steps = all_steps.at[0].set(x1)

        # solve from t1 to t0
        def euler_step(carry, i):
            x, all_x = carry
            t_i = t_array[i]
            drift_val = model.probability_flow_ode(
                x, t_i, guidance, pre_conds, post_conds
            )

            diffeq = model.diffeq

            dt = t_array[i - 1] - t_i
            x_next = x + drift_val * (-dt)
            x_next = guidance.apply_on_x_next(
                diffeq, x_next, t_i, pre_conds, post_conds, key=None
            )
            all_x = all_x.at[i].set(x_next)
            return (x_next, all_x), x_next  # ((new_x, all_x), output_for_scan)

        # initial carry contains both the current x and the array of all steps
        (_, all_steps), _ = jax.lax.scan(
            euler_step, (x1, all_steps), jnp.arange(1, num_steps + 1)
        )

        return all_steps


# # limamau: what is wrong here?
# class PredictorCorrectorSampler(AbstractSampler):
#     def __init__(
#         self,
#         t0: float = 1e-3,
#         t1: float = 1.0,
#         num_pred_steps: int = 4096,
#         num_correct_steps: int = 5,
#         tau: float = 0.25,
#     ):
#         self.t0 = jnp.array([t0])
#         self.t1 = jnp.array([t1])
#         self.num_pred_steps = num_pred_steps
#         self.num_correct_steps = num_correct_steps
#         self.tau = tau

#     @filter_jit
#     def single_sample(
#         self,
#         model: AbstractDiffusionModel,
#         data_shape: Tuple[int, ...],
#         guidance: AbstractGuidance,
#         pre_conds: Optional[Array],
#         post_conds: Optional[Array],
#         key: Key,
#     ) -> Array:
#         # prep
#         key, subkey = jr.split(key)
#         sigma_max = model.diffeq.sigma(self.t1)
#         x1 = jr.normal(subkey, data_shape) * sigma_max
#         dim_score = len(x1)
#         dt = (self.t1 - self.t0) / self.num_pred_steps

#         def score_fn(x, t):
#             return model.score(x, t, pre_conds, key=None)

#         def correct_step(carry, _):
#             x, t, key = carry
#             key, subkey = jr.split(key)
#             score = guidance.apply_on_score(
#                 score_fn, model.sde, x, t, pre_conds, post_conds, key=None
#             )
#             norm2 = jnp.mean(jnp.square(score))
#             delta = self.tau * dim_score / norm2
#             noise = jr.normal(subkey, data_shape)
#             x = x + delta * score + jnp.sqrt(2 * delta) * noise
#             return (x, t, key), None

#         def correct_scan(x, t, key):
#             (x_final, _, key_final), _ = jax.lax.scan(
#                 correct_step, (x, t, key), None, length=self.num_correct_steps
#             )
#             return x_final, key_final

#         def pred_step(carry, _):
#             x_curr, t_curr, key = carry
#             key, sub = jr.split(key)
#             t_next = t_curr - dt
#             mu_ratio = model.diffeq.mu(t_next) / model.diffeq.mu(t_curr)
#             sigma_ratio = model.diffeq.sigma(t_next) / model.diffeq.sigma(t_curr)
#             sigma2 = jnp.square(model.diffeq.sigma(t_curr))
#             score_curr = guidance.apply_on_score(
#                 score_fn, model.sde, x_curr, t_curr, pre_conds, post_conds, key=None
#             )
#             x_pred = mu_ratio * x_curr + (mu_ratio - sigma_ratio) * score_curr * sigma2

#             # corrector step
#             x_corr, key = correct_scan(x_pred, t_next, key)
#             x_corr = guidance.apply_on_x_next(
#                 model.sde, x_corr, t_next, pre_conds, post_conds, key=key
#             )
#             return (x_corr, t_next, key), None

#         # full scan
#         (x_final, t_next, _), _ = jax.lax.scan(
#             pred_step, (x1, self.t1, key), None, length=self.num_pred_steps
#         )
#         return x_final


class ODEDiffraxSampler(AbstractODESampler):
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
        self.dt0 = dt0
        self.t0 = t0
        self.t1 = t1
        self.solver = solver
        self.stepsize_controller = stepsize_controller

    @filter_jit
    def single_sample(
        self,
        model: AbstractModel,
        data_shape: Tuple[int, ...],
        guidance: AbstractGuidance,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        key: Key,
    ) -> Array:
        def fun(t, x, args):
            return model.probability_flow_ode(x, t, guidance, pre_conds, post_conds)

        term = dfx.ODETerm(fun)

        # solve from t1 to t0
        sigma_max = model.sigma_max
        x1 = jr.normal(key, data_shape) * sigma_max
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
