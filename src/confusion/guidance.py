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
        c: Array | None,
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
        c: Array | None,
        *,
        key: Key,
    ) -> Array:
        return model.score(x, t, c, key=key)


class MomentMatchingGuidance(AbstractGuidance):
    C: Array
    y: Array

    def __init__(self, const_matrix: ArrayLike, y: ArrayLike):
        self.C = jnp.asarray(const_matrix)
        self.y = jnp.asarray(y)

    # limamau:
    # (1) re-write this code as a function of the formula obtained in Tweedie’s Covariance
    # then it's easier to use this both for variance preserving and variance exploding
    # (2) that should also open up the possibility to have a ManifoldGuidance class by switching the score
    # to (y + mean) * std for the values that should be inferred during guidance - maybe use a list of bools
    # to encode that and y should be a tuple of arrays with the same dimension as the number of true values
    def apply(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Array | None,
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

        return eqx.filter_grad(logpdf)(x)
