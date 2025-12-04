from abc import abstractmethod
from typing import Callable, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, Key

from confusion.diffeqs.abstract import AbstractDiffEq


class AbstractGuidance:
    @abstractmethod
    def apply_on_score(
        self,
        score_fn: Callable[[Array, Array], Array],
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def apply_on_x_next(
        self,
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        raise NotImplementedError


# essentially no guidance at all
class GuidanceFree(AbstractGuidance):
    def __init__(self):
        pass

    def apply_on_score(
        self,
        score_fn: Callable[[Array, Array], Array],
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        return score_fn(x, t)

    def apply_on_x_next(
        self,
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        return x


class LinearFirstOrderMomentMatchingGuidance(AbstractGuidance):
    const_matrix: Array

    def __init__(
        self,
        const_matrix: ArrayLike,
    ):
        self.const_matrix = jnp.asarray(const_matrix)

    def apply_on_score(
        self,
        score_fn: Callable[[Array, Array], Array],
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        # check constraints shapes
        assert post_conds is not None
        # assert self.const_matrix.shape == (*post_conds.shape, *x.shape)

        # pre-calculations
        sigma_t2 = jnp.square(diffeq.sigma(t))
        mu_t = diffeq.mu(t)

        def x0_hat(x: Array) -> Array:
            return (x + sigma_t2 * score_fn(x, t)) / mu_t

        def log_pdf(x: Array) -> Array:
            loc = self.const_matrix @ x0_hat(x)
            scale = (
                sigma_t2 / jnp.square(mu_t) * self.const_matrix @ self.const_matrix.T
            )
            log_pdf = jax.scipy.stats.multivariate_normal.logpdf(
                post_conds, mean=loc, cov=scale
            )
            return jnp.squeeze(log_pdf)

        grad_logpdf = eqx.filter_grad(log_pdf)(x)

        # clipping
        score = score_fn(x, t)
        grad_logpdf = jnp.nan_to_num(grad_logpdf)

        result = score + grad_logpdf

        return result

    def apply_on_x_next(
        self,
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        return x


class FunctionFirstOrderMomentMatchingGuidance(AbstractGuidance):
    const_fn: Callable[[Array], Array]

    def __init__(
        self,
        const_fn: Callable[[Array], Array],
    ):
        self.const_fn = const_fn

    def apply_on_score(
        self,
        score_fn: Callable[[Array, Array], Array],
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        assert post_conds is not None

        sigma_t2 = jnp.square(diffeq.sigma(t))
        mu_t = diffeq.mu(t)

        # estimate x0
        def x0_hat_fn(x_: Array) -> Array:
            return (x_ + sigma_t2 * score_fn(x_, t)) / mu_t

        # flatten helper
        def flatten(z: Array):
            z_flat, _ = ravel_pytree(z)
            return z_flat

        # wrapped const_fn that works on original matrix shape
        def const_fn_wrapped(z_matrix: Array) -> Array:
            return flatten(self.const_fn(z_matrix))

        def log_pdf_fn(x_: Array) -> Array:
            # variable
            flat_post_conds = flatten(post_conds)

            # mean
            x0_hat = x0_hat_fn(x_)
            loc = const_fn_wrapped(x0_hat)

            # covariance
            gradC = jax.jacrev(const_fn_wrapped)(x0_hat)
            gradC_2d = gradC.reshape(gradC.shape[0], -1)
            scale = (sigma_t2 / jnp.square(mu_t)) * (gradC_2d @ gradC_2d.T)

            return jnp.squeeze(
                jax.scipy.stats.multivariate_normal.logpdf(
                    flat_post_conds, mean=loc, cov=scale
                )
            )

        # gradient wrt x
        grad_logpdf = eqx.filter_grad(log_pdf_fn)(x)

        # combine with model score
        score = score_fn(x, t)
        grad_logpdf = jnp.nan_to_num(grad_logpdf)

        return score + grad_logpdf

    def apply_on_x_next(
        self,
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        return x


class SecondOrderConstantMomentMatchingGuidance(AbstractGuidance):
    const_matrix: Array

    def __init__(
        self,
        const_matrix: ArrayLike,
        clipping_factor: float = 100.0,
    ):
        self.const_matrix = jnp.asarray(const_matrix)
        self.clipping_factor = clipping_factor

    def apply_on_score(
        self,
        score_fn: Callable[[Array, Array], Array],
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        assert post_conds is not None
        # assert self.const_matrix.shape == (*post_conds.shape, *x.shape)

        # pre-calculations
        sigma_t2 = jnp.square(diffeq.sigma(t))
        mu_t = diffeq.mu(t)

        def x0_hat(x: Array) -> Array:
            return (x + sigma_t2 * score_fn(x, t)) / mu_t

        # we follow the suggestions on the paper and compute this matrix
        # multiplication as the Jacobian of a function of x0_hat
        def mult_C_Sigma_hat(x: Array) -> Array:
            def fun(x: Array) -> Array:
                return sigma_t2 / mu_t * self.const_matrix @ x0_hat(x)

            # we use jacrev to compute the Jacobian because we can expect more inputs
            # (dimension of the signal) than outputs (dimension of the constraints) in
            # the map to be differentiated
            return eqx.filter_jacrev(fun)(x)  # pyright: ignore

        def log_pdf(x: Array) -> Array:
            loc = self.const_matrix @ x0_hat(x)
            # limamau: this is supposed to only work for specific dimensions...
            scale = mult_C_Sigma_hat(x) @ self.const_matrix.T
            log_pdf = jax.scipy.stats.multivariate_normal.logpdf(
                post_conds, mean=loc, cov=scale
            )
            return jnp.squeeze(log_pdf)

        grad_logpdf = eqx.filter_grad(log_pdf)(x)

        # clipping
        score = score_fn(x, t)
        norm_score = jnp.linalg.norm(score)
        grad_logpdf = jnp.nan_to_num(grad_logpdf)
        grad_logpdf = jnp.clip(
            grad_logpdf,
            -self.clipping_factor * norm_score,
            self.clipping_factor * norm_score,
        )

        return score + grad_logpdf

    def apply_on_x_next(
        self,
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        return x


# limamau: the idea is to use that as the a unified version of all the current
# sub-versions of the first order moment matching-based guidance
class FirstOrderMomentMatching(AbstractGuidance):
    constraint: Callable[[Array], Array]

    def __init__(
        self,
        constraint: Union[Array, Callable[[Array], Array]],
        cov_method: str = "approx",
        condition: str = "equality",
        clipping_factor: Optional[float] = None,
    ):
        assert cov_method in ["approx", "exact"], (
            "Invalid covariance method: {}".format(cov_method)
        )
        assert condition in ["equality", "inequality"], "Invalid condition: {}".format(
            condition
        )
        self.clipping_factor = clipping_factor

        # constraint
        if isinstance(constraint, Array):
            self.constraint = lambda x: constraint @ x
            # self.gradC = lambda x: self._flatten(constraint)
        elif isinstance(constraint, Callable):
            self.constraint = constraint
            # self.gradC = lambda x: jax.jacrev(constraint)(x)
        else:
            raise ValueError("Invalid constraint type: {}".format(type(constraint)))

        # condition
        if condition == "equality":
            self.condition = jax.scipy.stats.multivariate_normal.logpdf
        elif condition == "inequality":
            # limamau: currently that's only possible for single contraints
            # a general formula for the CDF of a multivariate Gaussian is non-trivial
            self.condition = jax.scipy.stats.norm.logcdf
        else:
            raise ValueError("Invalid condition type: {}".format(condition))

    @staticmethod
    def _flatten(z: Array):
        z_flat, _ = ravel_pytree(z)
        return z_flat

    def _constraint_flat_wrap(self, x: Array) -> Array:
        return self._flatten(self.constraint(x))

    def apply_on_score(
        self,
        score_fn: Callable[[Array, Array], Array],
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        assert post_conds is not None

        # pre-calculations
        sigma_t2 = jnp.square(diffeq.sigma(t))
        mu_t = diffeq.mu(t)

        def x0_hat_fn(x: Array) -> Array:
            return (x + sigma_t2 * score_fn(x, t)) / mu_t

        def sigma_hat_fn(x0_hat: Array) -> Array:
            gradC = jax.jacrev(self._constraint_flat_wrap)(x0_hat)
            gradC_2d = gradC.reshape(gradC.shape[0], -1)
            return gradC_2d @ gradC_2d.T

        def log_pdf_fn(x_: Array) -> Array:
            # variable
            flat_post_conds = self._flatten(post_conds)

            # mean
            x0_hat = x0_hat_fn(x_)
            loc = self._constraint_flat_wrap(x0_hat)

            # covariance
            cov = sigma_hat_fn(x0_hat)
            cov = (sigma_t2 / jnp.square(mu_t)) * cov

            return jnp.squeeze(self.condition(flat_post_conds, loc, cov))

        # gradient wrt x
        grad_logpdf = eqx.filter_grad(log_pdf_fn)(x)

        # combine with model score
        score = score_fn(x, t)
        grad_logpdf = jnp.nan_to_num(grad_logpdf)

        return score + grad_logpdf

    def apply_on_x_next(
        self,
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        return x


class SecondOrderMomentMatchingGuidance(AbstractGuidance):
    constraint: Callable[[Array], Array]

    def __init__(
        self,
        constraint: Union[Array, Callable[[Array], Array]],
        cov_method: str = "approx",
        condition: str = "equality",
        clipping_factor: float = 10.0,
    ):
        assert cov_method in ["approx", "exact"], (
            "Invalid covariance method: {}".format(cov_method)
        )
        assert condition in ["equality", "inequality"], "Invalid condition: {}".format(
            condition
        )
        self.clipping_factor = clipping_factor

        # constraint
        if isinstance(constraint, Array):
            self.constraint = lambda x: constraint @ x
            # self.gradC = lambda x: self._flatten(constraint)
        elif isinstance(constraint, Callable):
            self.constraint = constraint
            # self.gradC = lambda x: jax.jacrev(constraint)(x)
        else:
            raise ValueError("Invalid constraint type: {}".format(type(constraint)))

        # condition
        if condition == "equality":
            self.condition = jax.scipy.stats.multivariate_normal.logpdf
        elif condition == "inequality":
            # limamau: currently that's only possible for single contraints
            # a general formula for the CDF of a multivariate Gaussian is non-trivial
            self.condition = jax.scipy.stats.norm.logcdf
        else:
            raise ValueError("Invalid condition type: {}".format(condition))

    @staticmethod
    def _flatten(z: Array):
        z_flat, _ = ravel_pytree(z)
        return z_flat

    def _constraint_flat_wrap(self, x: Array) -> Array:
        return self._flatten(self.constraint(x))

    def apply_on_score(
        self,
        score_fn: Callable[[Array, Array], Array],
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        assert post_conds is not None

        # pre-calculations
        sigma_t2 = jnp.square(diffeq.sigma(t))
        mu_t = diffeq.mu(t)

        def x0_hat_fn(x: Array) -> Array:
            return (x + sigma_t2 * score_fn(x, t)) / mu_t

        # we follow the suggestions on the paper and compute this matrix
        # multiplication as the Jacobian of a function of x0_hat
        def mult_C_Sigma_hat_fn(x: Array) -> Array:
            return sigma_t2 / mu_t * self._constraint_flat_wrap(x0_hat_fn(x))

        def log_pdf_fn(x_: Array) -> Array:
            # variable
            flat_post_conds = self._flatten(post_conds)

            # mean
            x0_hat = x0_hat_fn(x_)
            loc = self._constraint_flat_wrap(x0_hat)

            # covariance: we use jacrev to compute the Jacobian because we can
            # expect more inputs (dimension of the signal) than outputs (dimension
            # of the constraints) in the map to be differentiated
            mult_C_Sigma_hat = eqx.filter_jacrev(mult_C_Sigma_hat_fn)(x)
            mult_C_Sigma_hat = mult_C_Sigma_hat.reshape(mult_C_Sigma_hat.shape[0], -1)  # pyright: ignore
            gradC = jax.jacrev(self._constraint_flat_wrap)(x0_hat)
            gradC_2d = gradC.reshape(gradC.shape[0], -1)
            cov = mult_C_Sigma_hat @ gradC_2d.T
            cov = (sigma_t2 / mu_t) * cov

            return jnp.squeeze(self.condition(flat_post_conds, loc, cov))

        # gradient wrt x
        grad_logpdf = eqx.filter_grad(log_pdf_fn)(x)

        # clipping
        score = score_fn(x, t)
        norm_score = jnp.linalg.norm(score)
        grad_logpdf = jnp.nan_to_num(grad_logpdf)
        grad_logpdf = jnp.clip(
            grad_logpdf,
            -self.clipping_factor * norm_score,
            self.clipping_factor * norm_score,
        )

        return score + grad_logpdf

    def apply_on_x_next(
        self,
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        return x


class ManifoldGuidance(AbstractGuidance):
    mask: Array

    def __init__(
        self,
        mask: ArrayLike,
    ):
        self.mask = jnp.asarray(mask)

    def apply_on_score(
        self,
        score_fn: Callable[[Array, Array], Array],
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        return score_fn(x, t)

    def apply_on_x_next(
        self,
        diffeq: AbstractDiffEq,
        x: Array,
        t: Array,
        pre_conds: Optional[Array],
        post_conds: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        # move y to the mean trajectory
        y = post_conds * diffeq.mu(t)

        # change values on x according to mask
        x = jnp.where(self.mask, y, x)

        return x
