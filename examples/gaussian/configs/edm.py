import os

import jax.numpy as jnp
import jax.random as jr
import optax

from confusion.diffeqs.sdes import VarianceExploding
from confusion.guidance import SecondOrderMomentMatchingGuidance
from confusion.models.diffusion import DenoiserDiffusionModel
from confusion.networks import MultiLayerPerceptron
from confusion.sampling import ConstantStepEulerMaruyamaSampler
from confusion.schedules import LinearTimeSchedule
from confusion.weighting import DenoiserWeighting


class Config:
    """Configuration for Elucidating the Design Space of Diffusion-Based Generative Models (EDM)."""

    name = "edm"

    # 1. keys
    seed = 5678
    key = jr.PRNGKey(seed)
    net_key, train_key, evaluate_key = jr.split(key, 3)

    # 2. dataset
    num_samples = 10_000

    # 3. network
    num_variables = 2
    hidden_size = 64
    proj_size = 128
    proj_scale = 1.0
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
    sigma_min = 0.01
    sigma_max = 10.0
    is_approximate = False
    sde = VarianceExploding(
        sigma_min,
        sigma_max,
        is_approximate=is_approximate,
    )

    # 5. diffusion model
    sigma_data = 0.5
    weighting = DenoiserWeighting(sde, sigma_data)
    model = DenoiserDiffusionModel(network, weighting, sde, sigma_data)

    # 6. optimization
    t0_training = 1e-5
    t1 = 1.0
    num_steps = 10_000
    lr = 1e-3
    train_batch_size = 512
    opt = optax.adam(lr)

    # 7. logging, evaluating and checkpointing
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
    time_schedule = LinearTimeSchedule()
    t0_sampling = 1e-3
    num_sampling_steps = 1000
    dt0 = (t1 - t0_sampling) / num_sampling_steps
    sample_size = 1000
    conds = None
    sampler = ConstantStepEulerMaruyamaSampler(dt0, t0_sampling, t1)

    # 9. guidance
    a = 1.0
    y = jnp.array([a])
    const_matrix = jnp.array([[1.0, 0.0]])
    guidance = SecondOrderMomentMatchingGuidance(const_matrix)
