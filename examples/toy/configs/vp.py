import os

import jax.numpy as jnp
import jax.random as jr
import optax

from confusion.diffusion import VariancePreserving
from confusion.guidance import GuidanceFree, MomentMatchingGuidance
from confusion.networks import MultiLayerPerceptron
from confusion.sampling import EulerMaruyamaSampler


def get_weight2_fn(beta_min_bar, beta_max_bar):
    return lambda t: 1 - jnp.exp(
        -0.5 * t**2 * (beta_max_bar - beta_min_bar) - t * beta_min_bar
    )


class Config:
    """Configuration for Variance Preserving."""

    name = "vp"

    # 1. keys
    seed = 5678
    key = jr.PRNGKey(seed)
    data_key, net_key, train_key, sample_key = jr.split(key, 4)

    # 2. dataset
    num_samples = 10_000

    # 3. network
    num_variables = 2
    hidden_size = 256
    is_conditional = False
    network = MultiLayerPerceptron(
        num_variables,
        hidden_size,
        key=net_key,
        is_conditional=is_conditional,
    )

    # 4. diffusion model
    t0 = 0.1
    t1 = 3.0
    beta_min_bar = 0.1
    beta_max_bar = 0.12
    weight2_fn = get_weight2_fn(beta_min_bar, beta_max_bar)
    model = VariancePreserving(
        network,
        weight2_fn,
        beta_min_bar,
        beta_max_bar,
    )

    # 5. optimization
    num_steps = 10_000
    lr = 1e-3
    batch_size = 16
    opt = optax.adam(lr)

    # 6. logging and checkpointing
    print_every = 1000
    max_save_to_keep = 1
    save_every = 5000
    saving_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../checkpoints/{name}",
    )

    # 7. sampling
    dt0 = 0.01
    sample_size = 1000
    conds = None
    sampler = EulerMaruyamaSampler(dt0, t0=t0, t1=t1)

    # 8. guidance
    # 8.1 no guidance
    guidance_free = GuidanceFree()
    # 8.2 moment matching
    do_A = 1.0
    const_matrix = jnp.array([[do_A, 0.0]])
    y = jnp.array([do_A])
    guidance = MomentMatchingGuidance(
        const_matrix,
        y,
    )
