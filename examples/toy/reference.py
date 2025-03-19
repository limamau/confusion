from typing import Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key


def get_joint(
    num_samples: int, key: Key, do_A: Optional[float] = None
) -> Tuple[Array, Array]:
    # split keys
    key_A, key_B, key_C = jr.split(key, 3)

    # A
    if do_A is None:
        A = jr.normal(key_A, (num_samples, 1))
    else:
        A = do_A * jnp.ones((num_samples, 1))

    # B
    B = A + 0.5 * jr.normal(key_B, (num_samples, 1))

    return A, B
