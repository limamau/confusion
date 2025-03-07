import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key


class GaussianFourierProjection(eqx.Module):
    gaussian: jax.Array

    def __init__(self, mapping_dim: int, scale: float = 10.0, *, key: Key):
        self.gaussian = jax.random.normal(key, (mapping_dim // 2,)) * scale

    def __call__(self, t: Array) -> Array:
        projection = t * self.gaussian * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(projection), jnp.cos(projection)], axis=-1).T
