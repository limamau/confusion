import os

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import optax

from confusion.diffusion import VarianceExploding
from confusion.guidance import MomentMatchingGuidance
from confusion.networks import CausalMultiLayerPerceptron
from confusion.sampling import ODESampler
from confusion.schedules import get_edm_sampling_ts


def get_weight2_fn(sigma_min, sigma_max):
    return lambda t: sigma_min * jnp.pow((sigma_max / sigma_min), 2 * t)


class Config:
    """Configuration for Causal Variance Exploding."""

    name = "cve"

    # 1. keys
    seed = 5678
    key = jr.PRNGKey(seed)
    data_key, net_key, train_key, sample_key = jr.split(key, 4)

    # 2. dataset
    num_samples = 10_000

    # 3. network
    num_variables = 3
    num_blocks = 3
    hidden_dim = 256
    temb_dim = 2
    projection_scale = 1.0
    num_heads = 1
    qkv_size = 8
    is_conditional = False
    causal_mask = jnp.ones((num_variables, num_variables), dtype=bool)
    network = CausalMultiLayerPerceptron(
        num_blocks=num_blocks,
        vars_dim=num_variables,
        hidden_dim=hidden_dim,
        temb_dim=temb_dim,
        projection_scale=projection_scale,
        causal_mask=causal_mask,
        num_heads=num_heads,
        qkv_size=qkv_size,
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
    std_sampler = ODESampler(dt0, t0=t0, t1=t1)
    N = 500
    ts = get_edm_sampling_ts(model, N=N, t1=t1, t0=t0)
    step_size_controller = dfx.StepTo(ts)
    edm_sampler = ODESampler(
        None, t0=t0, t1=t1, solver=dfx.Heun(), step_size_controller=step_size_controller
    )

    # 8. guidance
    do_B = 1.0
    const_matrix = jnp.array([[0.0, do_B, 0.0]])
    y = jnp.array([do_B])
    moment_matching_guidance = MomentMatchingGuidance(
        const_matrix,
        y,
    )
