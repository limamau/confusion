from abc import abstractmethod
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .diffusion import AbstractDiffusionModel


class AbstractGuidance:
    @abstractmethod
    def apply_on_score(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def apply_on_x_next(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
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
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        return model.score(x, t, c, key=key)

    def apply_on_x_next(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        return x


# following https://proceedings.mlr.press/v202/finzi23a.html
class MomentMatchingGuidance(AbstractGuidance):
    const_matrix: Array
    y: Array

    def __init__(
        self,
        const_matrix: ArrayLike,
        y: ArrayLike,
        clipping_factor: float = 10.0,
    ):
        self.const_matrix = jnp.asarray(const_matrix)
        self.y = jnp.asarray(y)
        self.clipping_factor = clipping_factor

    def apply_on_score(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        # check constraints shapes
        assert self.const_matrix.shape == (*self.y.shape, *x.shape)

        # pre-calculations
        sigma_t2 = jnp.square(model.sde.sigma(t))
        s_t = model.sde.s(t)

        def x0_hat(x: Array) -> Array:
            return (x + sigma_t2 * model.score(x, t, c, key=key)) / s_t

        # we follow the suggestions on the paper and compute this matrix
        # multiplication as the Jacobian of a function of x0_hat
        def mult_C_Sigma_hat(x: Array) -> Array:
            def fun(x: Array) -> Array:
                return sigma_t2 / s_t * self.const_matrix @ x0_hat(x)

            # we use jacrev to compute the Jacobian because we can expect more inputs
            # (dimension of the signal) than outputs (dimension of the constraints) in
            # the map to be differentiated
            return eqx.filter_jacrev(fun)(x)  # pyright: ignore

        def log_pdf(x: Array) -> Array:
            loc = self.const_matrix @ x0_hat(x)
            scale = mult_C_Sigma_hat(x) @ self.const_matrix.T
            log_pdf = jax.scipy.stats.multivariate_normal.logpdf(
                self.y, mean=loc, cov=scale
            )
            return jnp.squeeze(log_pdf)

        grad_logpdf = eqx.filter_grad(log_pdf)(x)

        # clipping
        score = model.score(x, t, c, key=key)
        norm_score = jnp.linalg.norm(score)
        grad_logpdf = jnp.nan_to_num(grad_logpdf)
        grad_logpdf = jnp.clip(
            grad_logpdf,
            -self.clipping_factor * norm_score,
            self.clipping_factor * norm_score,
        )

        result = score + grad_logpdf

        return result

    def apply_on_x_next(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        return x


class ManifoldGuidance(AbstractGuidance):
    mask: Array
    y: Array

    def __init__(
        self,
        mask: ArrayLike,
        y: ArrayLike,
    ):
        self.mask = jnp.asarray(mask)
        self.y = jnp.asarray(y)

    def apply_on_score(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        return model.score(x, t, c, key=key)

    def apply_on_x_next(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        # move y to the mean trajectory
        y = self.y * model.sde.s(t)

        # change values on x according to mask
        x = jnp.where(self.mask, y, x)

        return x


class MixedGuidance(AbstractGuidance):
    const_matrix: Array
    moment_matching_y: Array
    mask: Array
    manifold_y: Array

    def __init__(
        self,
        const_matrix: ArrayLike,
        moment_matching_y: ArrayLike,
        mask: ArrayLike,
        manifold_y: ArrayLike,
    ):
        self.const_matrix = jnp.asarray(const_matrix)
        self.moment_matching_y = jnp.asarray(moment_matching_y)
        self.mask = jnp.asarray(mask)
        self.manifold_y = jnp.asarray(manifold_y)

    def apply_on_score(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        # check constraints shapes
        assert self.const_matrix.shape == (*self.moment_matching_y.shape, *x.shape)

        # pre-calculations
        sigma_t2 = jnp.square(model.sde.sigma(t))
        s_t = model.sde.s(t)
        score = model.score(x, t, c, key=key)

        def x0_hat(x: Array) -> Array:
            return (x + sigma_t2 * model.score(x, t, c, key=key)) / s_t

        # we follow the suggestions on the paper and compute this matrix
        # multiplication as the Jacobian of a function of x0_hat
        def mult_C_Sigma_hat(x: Array) -> Array:
            def fun(x: Array) -> Array:
                return sigma_t2 / s_t * self.const_matrix @ x0_hat(x)

            # we use jacrev to compute the Jacobian because we can expect more inputs
            # (dimension of the signal) than outputs (dimension of the constraints) in
            # the map to be differentiated
            return eqx.filter_jacrev(fun)(x)  # pyright: ignore

        def log_pdf(x: Array) -> Array:
            loc = self.const_matrix @ x0_hat(x)
            scale = mult_C_Sigma_hat(x) @ self.const_matrix.T
            log_pdf = jax.scipy.stats.multivariate_normal.logpdf(
                self.moment_matching_y, mean=loc, cov=scale
            )
            return jnp.squeeze(log_pdf)

        grad_logpdf = eqx.filter_grad(log_pdf)(x)

        result = score + grad_logpdf

        return result

    def apply_on_x_next(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        # move y to the mean trajectory
        y = self.manifold_y * model.sde.s(t)

        # change values on x according to mask
        x = jnp.where(self.mask, y, x)

        return x


# leaving that here...
# def log_pdf(x: Array) -> Array:
#     loc = self.C @ x0_hat(x)
#     scale_matrix = mult_C_Sigma_hat(x) @ self.C.T
#     k = loc.shape[0]
#     diff = y - loc
#     sign, logdet = jnp.linalg.slogdet(scale_matrix)
#     inverse = jnp.linalg.inv(scale_matrix)
#     log_pdf = -0.5 * (
#         k * jnp.log(2 * jnp.pi) + logdet + diff.T @ inverse @ diff
#     )
#     return jnp.squeeze(log_pdf)

# def log_pdf(x: Array) -> Array:
#     loc = self.C @ x0_hat(x)
#     diff = y - loc
#     scale = mult_C_Sigma_hat(x) @ self.C.T
#     k = loc.shape[0]
#     det = jnp.linalg.det(scale)
#     inv = jnp.linalg.inv(scale)
#     den = jnp.sqrt(jnp.power(2 * jnp.pi, k) * det)
#     num = jnp.exp(-0.5 * diff.T @ inv @ diff)
#     pdf = num / den
#     log_pdf = jnp.log(pdf)
#     return jnp.squeeze(log_pdf)
