import os

import jax.numpy as jnp
import jax.random as jr
import optax

from confusion.diffusion import StandardDiffusionModel
from confusion.guidance import MomentMatchingGuidance
from confusion.losses import ScoreMatchingLoss, StandardWeighting
from confusion.networks import MultiLayerPerceptron
from confusion.sampling import EulerMaruyamaSampler
from confusion.sdes import VarianceExploding


class Config:
    """Configuration for Variance Exploding."""

    name = "ve"

    # 1. keys
    seed = 5678
    key = jr.PRNGKey(seed)
    net_key, train_key, evaluate_key = jr.split(key, 3)

    # 2. dataset
    num_samples = 10_000

    # 3. network
    num_variables = 2
    hidden_size = 256
    proj_size = 32
    proj_scale = 5.0
    is_conditional = False
    network = MultiLayerPerceptron(
        proj_size,
        proj_scale,
        num_variables,
        hidden_size,
        key=net_key,
        is_conditional=is_conditional,
    )

    # 4. sde
    t0 = 0.1
    t1 = 1.0
    sigma_min = 0.1
    sigma_max = 2.0
    is_approximate = False
    sde = VarianceExploding(
        sigma_min,
        sigma_max,
        is_approximate=is_approximate,
    )

    # 5. diffusion model
    sigma_data = 1.0  # for completeness, but not used
    model = StandardDiffusionModel(network, sde)

    # 6. optimization
    num_steps = 10_000
    lr = 1e-3
    train_batch_size = 32
    opt = optax.adam(lr)
    weighting = StandardWeighting(sde)
    loss = ScoreMatchingLoss(weighting)

    # 7. logging and checkpointing
    print_loss_every = 1000
    max_save_to_keep = 1
    save_every = 5000
    saving_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../checkpoints/{name}",
    )
    eval_batch_size = 512
    eval_every = 5000

    # 8. sampling
    dt0 = 0.005
    sample_size = 1000
    conds = None
    sampler = EulerMaruyamaSampler(dt0, t0=t0, t1=t1)

    # 9. guidance
    do_A = 1.0
    const_matrix = jnp.array([[do_A, 0.0]])
    y = jnp.array([do_A])
    guidance = MomentMatchingGuidance(
        const_matrix,
        y,
    )
