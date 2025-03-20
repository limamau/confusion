import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key


class GaussianFourierProjection(eqx.Module):
    gaussian: jax.Array

    def __init__(self, proj_size: int, proj_scale: float, *, key: Key):
        self.gaussian = jax.random.normal(key, (proj_size // 2,)) * proj_scale

    def __call__(self, t: Array) -> Array:
        projection = t * self.gaussian * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(projection), jnp.cos(projection)], axis=-1).T
