import os

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import optax

from confusion.diffusion import VariancePreserving
from confusion.guidance import GuidanceFree, ManifoldGuidance, MomentMatchingGuidance
from confusion.networks import CausalMultiLayerPerceptron
from confusion.sampling import EulerMaruyamaSampler, ODESampler
from confusion.schedules import get_edm_sampling_ts


def get_weight2_fn(beta_min_bar, beta_max_bar):
    return lambda t: 1 - jnp.exp(
        -0.5 * t**2 * (beta_max_bar - beta_min_bar) - t * beta_min_bar
    )


class Config:
    """Configuration for Causal Variance Preserving."""

    name = "cvp-all"

    # 1. keys
    seed = 5678
    key = jr.PRNGKey(seed)
    data_key, net_key, train_key, sample_key = jr.split(key, 4)

    # 2. dataset
    num_samples = 10_000

    # 3. network
    num_variables = 3
    num_blocks = 3
    hidden_size = 256
    temb_size = 2
    projection_scale = 1.0
    num_heads = 1
    qkv_size = 8
    is_conditional = False
    use_shared_linears = False
    causal_mask = jnp.ones((num_variables, num_variables), dtype=bool)
    # causal_mask = jnp.zeros((num_variables, num_variables), dtype=bool)
    # causal_mask = causal_mask.at[0, 0].set(True)
    # causal_mask = causal_mask.at[1, 1].set(True)
    # causal_mask = causal_mask.at[2, 2].set(True)
    # causal_mask = causal_mask.at[0, 1].set(True)
    # causal_mask = causal_mask.at[1, 2].set(True)
    network = CausalMultiLayerPerceptron(
        num_blocks=num_blocks,
        vars_size=num_variables,
        hidden_size=hidden_size,
        temb_size=temb_size,
        projection_scale=projection_scale,
        causal_mask=causal_mask,
        num_heads=num_heads,
        qkv_size=qkv_size,
        key=net_key,
        is_conditional=is_conditional,
        use_shared_linears=use_shared_linears,
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
    # 7.1 Euler-Maruyama sampler
    dt0 = 0.01
    sample_size = 1000
    conds = None
    em_sampler = EulerMaruyamaSampler(dt0, t0=t0, t1=t1)
    # 7.2 EDM sampler
    N = 500
    ts = get_edm_sampling_ts(model, N=N, t1=t1, t0=t0)
    stepsize_controller = dfx.StepTo(ts)
    edm_sampler = ODESampler(
        None, t0=t0, t1=t1, solver=dfx.Heun(), stepsize_controller=stepsize_controller
    )

    # 8. guidance
    # 8.1 no guidance
    guidance_free = GuidanceFree()
    # 8.2 moment matching
    do_B = 1.0
    const_matrix = jnp.array([[0.0, do_B, 0.0]])
    moment_matching_y = jnp.array([do_B])
    moment_matching_guidance = MomentMatchingGuidance(
        const_matrix,
        moment_matching_y,
    )
    # 8.3 manifold
    mask = jnp.array([False, True, False])
    manifold_y = jnp.array([0.0, do_B, 0.0])
    manifold_guidance = ManifoldGuidance(mask, manifold_y)
