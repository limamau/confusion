from abc import abstractmethod
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .diffusion import AbstractDiffusionModel, VarianceExploding


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

    def apply(
        self,
        model: AbstractDiffusionModel,
        x: Array,
        t: Array,
        c: Array | None,
        *,
        key: Optional[Key] = None,
    ) -> Array:
        # that only works for variance exploding (for now)
        assert isinstance(model, VarianceExploding)

        # pre-calculations of values independent of x
        sigma_t2 = jnp.square(model.sigma(t))

        def x0_hat(x: Array) -> Array:
            score = model.score(x, t, c, key=key)
            return x + sigma_t2 * score

        # we follow the suggestions on the paper and compute this matrix
        # multiplication as the Jacobian of a function of x0_hat
        def mult_C_Sigma_hat(x: Array) -> Array:
            def fun(x: Array):
                return sigma_t2 * self.C @ model.score(x, t, c, key=key)

            # we use jacrev to compute the Jacobian because we can expect more inputs
            # (dimension of the signal) than outputs (dimension of the constraints)
            return eqx.filter_jacrev(fun)(x)  # pyright: ignore

        def logpdf(x: Array) -> Array:
            return jax.scipy.stats.norm.logpdf(
                self.y,
                loc=self.C @ x0_hat(x),
                scale=mult_C_Sigma_hat(x) @ self.C.T,
            )[0, 0]  # this works because t is a float for sampling,
            # but it could be nice to add batch dimension in sampling

        return eqx.filter_grad(logpdf)(x)
