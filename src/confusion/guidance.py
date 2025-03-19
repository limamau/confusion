from abc import abstractmethod
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .diffusion import AbstractDiffusionModel


class AbstractGuidance:
    @abstractmethod
    def apply(
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

    def apply(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Key,
    ) -> Array:
        return model.score(x, t, c, key=key)


# following https://proceedings.mlr.press/v202/finzi23a.html
class MomentMatchingGuidance(AbstractGuidance):
    C: Array
    y: Array

    def __init__(self, const_matrix: ArrayLike, y: ArrayLike):
        self.C = jnp.asarray(const_matrix)
        self.y = jnp.asarray(y)

    def apply(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        # check constraints shapes
        assert self.C.shape == (*self.y.shape, *x.shape)

        # pre-calculations
        sigma_t2 = jnp.square(model.sigma(t))
        s_t = model.s(t)

        def x0_hat(x: Array) -> Array:
            score = model.score(x, t, c, key=key)
            return (x + sigma_t2 * score) / s_t

        # we follow the suggestions on the paper and compute this matrix
        # multiplication as the Jacobian of a function of x0_hat
        def mult_C_Sigma_hat(x: Array) -> Array:
            def fun(x: Array):
                return sigma_t2 * self.C @ model.score(x, t, c, key=key) / s_t

            # we use jacrev to compute the Jacobian because we can expect more inputs
            # (dimension of the signal) than outputs (dimension of the constraints) in
            # the map to be differentiated
            return eqx.filter_jacrev(fun)(x)  # pyright: ignore

        def logpdf(x: Array) -> Array:
            return jax.scipy.stats.norm.logpdf(
                self.y,
                loc=self.C @ x0_hat(x),
                scale=mult_C_Sigma_hat(x) @ self.C.T,
            )[0, 0]  # t is a float for sampling

        return eqx.filter_grad(logpdf)(x) + model.score(x, t, c, key=key)


class ManifoldGuidance:
    mask: Array
    y: Array

    def __init__(self, mask: ArrayLike, y: ArrayLike):
        self.mask = jnp.asarray(mask)
        self.y = jnp.asarray(y)

    def apply(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        # original score
        original_score = model.score(x, t, c, key=key)

        # perturbation of reference values
        # to the same noise level at t
        mean, std = model.perturbation(self.y, t)
        perturbed_y = mean + self.y * std

        # change values on score according to mask
        modified_score = jnp.where(self.mask, perturbed_y, original_score)

        return modified_score
