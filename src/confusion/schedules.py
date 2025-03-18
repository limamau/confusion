import jax.numpy as jnp

from .diffusion import AbstractDiffusionModel


def get_edm_sampling_ts(
    model: AbstractDiffusionModel,
    rho: float = 7.0,
    N: int = 300,
    t0: float = 1e-3,
    t1: float = 1.0,
):
    sigma_max = model.sigma(jnp.array([t1]))
    sigma_min = model.sigma(jnp.array([t0]))

    sigmas = jnp.power(
        jnp.power(sigma_max, (1 / rho))
        + jnp.arange(N)
        / (N - 1)
        * (jnp.power(sigma_min, (1 / rho)) - jnp.power(sigma_max, (1 / rho))),
        rho,
    )

    ts = jnp.array([model.t(sigma_i) for sigma_i in sigmas])
    ts = jnp.clip(ts, t0, t1)

    # to bypass dfx.diffeqsolve's assertion
    if abs(ts[0] - t1) < 1e-2:
        ts = jnp.concatenate([jnp.array([t1]), ts[1:]])

    return ts
