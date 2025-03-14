import os

import jax.numpy as jnp
import jax.random as jr
import optax

from confusion.diffusion import VariancePreserving
from confusion.guidance import MomentMatchingGuidance
from confusion.networks import MultiLayerPerceptron
from confusion.sampling import ODESampler


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
    num_variables = 3
    hidden_size = 256
    is_conditional = False
    t1 = 3.0
    network = MultiLayerPerceptron(
        num_variables,
        hidden_size,
        t1,
        key=net_key,
        is_conditional=is_conditional,
    )

    # 4. diffusion model
    t0 = 0.1

    @staticmethod
    def int_beta_fn(t):
        return t

    @staticmethod
    def weight_fn(t):
        return 1 - jnp.exp(-Config.int_beta_fn(t))

    model = VariancePreserving(
        network,
        weight_fn,
        t0,
        t1,
        int_beta_fn,
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
    sampler = ODESampler(dt0, t1)

    # 8. guidance
    do_B = 1.0
    const_matrix = jnp.array([[0.0, do_B, 0.0]])
    y = jnp.array([do_B])
    moment_matching_guidance = MomentMatchingGuidance(
        const_matrix,
        y,
    )
