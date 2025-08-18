import os

import jax.random as jr
import optax

from confusion.diffusion import StandardDiffusionModel
from confusion.networks import UNet1D
from confusion.sampling import ConstantStepEulerMaruyamaSampler
from confusion.schedules import LinearTimeSchedule
from confusion.sdes import VarianceExploding
from confusion.weighting import SquaredWeighting


class Config:
    """Configuration for Variance Exploding."""

    name = "ve"

    # 1. keys
    seed = 5678
    key = jr.PRNGKey(seed)
    net_key, train_key, evaluate_key = jr.split(key, 3)

    # 2. dataset
    num_samples = 1_000
    seq_len = 32

    # 3. network
    network = UNet1D(
        data_shape=(2, seq_len),
        proj_size=32,
        proj_scale=1.0,
        is_biggan=False,
        dim_mults=[1, 2, 4],
        hidden_size=16,
        heads=2,
        dim_head=8,
        dropout_rate=0.1,
        num_res_blocks=2,
        attn_resolutions=[],
        key=net_key,
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
    sigma_data = 1.0  # for completeness, but not used
    weighting = SquaredWeighting(sde)
    model = StandardDiffusionModel(network, weighting, sde)

    # 6. optimization
    t0_training = 1e-5
    t1 = 1.0
    num_steps = 10_000
    lr = 1e-3
    train_batch_size = 16
    opt = optax.adam(lr)

    # 7. logging and checkpointing
    print_loss_every = 500
    max_save_to_keep = 1
    save_every = 1000
    saving_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../checkpoints/{name}",
    )
    eval_batch_size = 16
    eval_every = 1000

    # 8. sampling
    t0_sampling = 1e-3
    time_schedule = LinearTimeSchedule()
    dt0 = 0.001
    sample_size = 100
    conds = None
    sampler = ConstantStepEulerMaruyamaSampler(dt0, t0=t0_sampling, t1=t1)
