from abc import abstractmethod
from typing import Callable, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, Key

from confusion.diffeqs.abstract import AbstractDiffEq
from confusion.diffeqs.sdes import AbstractSDE


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
    def apply_on_diffusion(
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


class GuidanceFree(AbstractGuidance):
    """Essentially no guidance at all."""

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

    def apply_on_diffusion(
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
        assert isinstance(diffeq, AbstractSDE)
        return diffeq.diffusion(t)

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


class FirstOrderMomentMatching(GuidanceFree):
    """Moment matching guidance assuming the Jacobian of the score to be zero."""

    constraint: Callable[[Array], Array]

    def __init__(
        self,
        constraint: Union[Array, Callable[[Array], Array]],
        condition: str = "equality",
    ):
        assert condition in ["equality", "inequality"], "Invalid condition: {}".format(
            condition
        )

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

        def log_pdf_fn(x_: Array) -> Array:
            # variable
            flat_post_conds = self._flatten(post_conds)

            # mean
            x0_hat = x0_hat_fn(x_)
            loc = self._constraint_flat_wrap(x0_hat)

            # covariance
            grad_x0_hat = jax.jacrev(x0_hat_fn)(x_)
            grad_x0_hat_2d = grad_x0_hat.reshape(grad_x0_hat.shape[0], -1)  # pyright: ignore
            Sigma_hat = (sigma_t2 / mu_t) * grad_x0_hat_2d

            gradC = jax.jacrev(self._constraint_flat_wrap)(x0_hat)
            gradC_2d = gradC.reshape(gradC.shape[0], -1)

            cov = gradC_2d @ Sigma_hat @ gradC_2d.T
            cov = jax.lax.stop_gradient(cov)

            return jnp.squeeze(self.condition(flat_post_conds, loc, cov))

        # gradient wrt x
        grad_logpdf = eqx.filter_grad(log_pdf_fn)(x)

        # combine with model score
        score = score_fn(x, t)
        grad_logpdf = jnp.nan_to_num(grad_logpdf)

        return score + grad_logpdf


class SecondOrderMomentMatching(GuidanceFree):
    """Second order moment matching guidance."""

    constraint: Callable[[Array], Array]

    def __init__(
        self,
        constraint: Union[Array, Callable[[Array], Array]],
        condition: str = "equality",
    ):
        assert condition in ["equality", "inequality"], "Invalid condition: {}".format(
            condition
        )

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

        def log_pdf_fn(x_: Array) -> Array:
            # variable
            flat_post_conds = self._flatten(post_conds)

            # mean
            x0_hat = x0_hat_fn(x_)
            loc = self._constraint_flat_wrap(x0_hat)
            grad_x0_hat = jax.jacrev(x0_hat_fn)(x_)
            grad_x0_hat_2d = grad_x0_hat.reshape(grad_x0_hat.shape[0], -1)  # pyright: ignore
            Sigma_hat = (sigma_t2 / mu_t) * grad_x0_hat_2d

            # covariance
            gradC = jax.jacrev(self._constraint_flat_wrap)(x0_hat)
            gradC_2d = gradC.reshape(gradC.shape[0], -1)

            cov = gradC_2d @ Sigma_hat @ gradC_2d.T
            cov = jax.lax.stop_gradient(cov)

            return jnp.squeeze(self.condition(flat_post_conds, loc, cov))

        # gradient wrt x
        grad_logpdf = eqx.filter_grad(log_pdf_fn)(x)

        score = score_fn(x, t)
        grad_logpdf = jnp.nan_to_num(grad_logpdf)

        return score + grad_logpdf


class ManifoldGuidance(GuidanceFree):
    """Manifold guidance is basically moving things manually to the conditions."""

    mask: Array

    def __init__(
        self,
        mask: ArrayLike,
    ):
        self.mask = jnp.asarray(mask)

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
        assert post_conds is not None

        # move y to the mean trajectory
        y = post_conds * diffeq.mu(t)

        # change values on x according to mask
        x = jnp.where(self.mask, y, x)

        return x
