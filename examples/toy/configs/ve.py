import os

import jax.numpy as jnp
import jax.random as jr
import optax

from confusion.diffusion import VarianceExploding
from confusion.guidance import MomentMatchingGuidance
from confusion.networks import MultiLayerPerceptron
from confusion.sampling import EulerMaruyamaSampler


def get_weight2_fn(sigma_min, sigma_max):
    return lambda t: sigma_min * jnp.pow((sigma_max / sigma_min), 2 * t)


class Config:
    """Configuration for Variance Exploding."""

    name = "ve"

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
    sigma_min = 0.1
    sigma_max = 0.12
    is_approximate = False
    weight_fn = get_weight2_fn(sigma_min, sigma_max)
    model = VarianceExploding(
        network,
        weight_fn,
        sigma_min,
        sigma_max,
        is_approximate=is_approximate,
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
    do_A = 1.0
    const_matrix = jnp.array([[do_A, 0.0]])
    y = jnp.array([do_A])
    guidance = MomentMatchingGuidance(
        const_matrix,
        y,
    )
