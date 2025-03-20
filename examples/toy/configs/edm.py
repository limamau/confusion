import os

import jax.numpy as jnp
import jax.random as jr
import optax

from confusion.diffusion import EDMDiffusionModel
from confusion.guidance import MomentMatchingGuidance
from confusion.losses import EDMWeighting, ScoreMatchingLoss
from confusion.networks import MultiLayerPerceptron
from confusion.sampling import EulerMaruyamaSampler, edm_sampling_ts
from confusion.sdes import VarianceExploding


class Config:
    """Configuration for Elucidating the Design Space of Diffusion-Based Generative Models (EDM)."""

    name = "edm"

    # 1. keys
    seed = 5678
    key = jr.PRNGKey(seed)
    data_key, net_key, train_key, sample_key = jr.split(key, 4)

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
    sigma_max = 0.5
    is_approximate = False
    sde = VarianceExploding(
        sigma_min,
        sigma_max,
        is_approximate=is_approximate,
    )

    # 5. diffusion model
    sigma_data = 0.5
    model = EDMDiffusionModel(network, sde, sigma_data)

    # 6. optimization
    num_steps = 10_000
    lr = 1e-3
    batch_size = 32
    opt = optax.adam(lr)
    weighting = EDMWeighting(sde, sigma_data)
    loss = ScoreMatchingLoss(weighting, t0=t0, t1=t1)

    # 7. logging and checkpointing
    print_every = 1000
    max_save_to_keep = 1
    save_every = 5000
    saving_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../checkpoints/{name}",
    )

    # 8. sampling
    dt0 = 0.05
    sample_size = 1000
    conds = None
    edm_ts = edm_sampling_ts(sde, t0=t0, t1=t1)
    sampler = EulerMaruyamaSampler(
        dt0, t0, t1
    )  # limamau: implement a working EDM sampler

    # 9. guidance
    do_A = 1.0
    const_matrix = jnp.array([[do_A, 0.0]])
    y = jnp.array([do_A])
    guidance = MomentMatchingGuidance(
        const_matrix,
        y,
    )
