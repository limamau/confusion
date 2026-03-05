import os

import jax.numpy as jnp
import jax.random as jr
import optax

from confusion.diffeqs.odes import OTFlowMatching
from confusion.guidance import ManifoldGuidance
from confusion.models.flow import StandardFlowMatching
from confusion.networks import MultiLayerPerceptron
from confusion.sampling import ScheduledEulerSampler
from confusion.schedules import LinearTimeSchedule


class Config:
    """Configuration for Flow Matching with Optimal Transport ODE."""

    name = "flow"

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

    # 4. ode
    sigma_min = 1e-5
    ode = OTFlowMatching(sigma_min=sigma_min)

    # 5. flow matching model
    sigma_data = 1.0
    model = StandardFlowMatching(network, ode)

    # 6. optimization
    t0_training = 1e-5
    t1 = 1.0
    num_steps = 10_000
    lr = 1e-3
    train_batch_size = 512
    opt = optax.adam(lr)

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
    t0_sampling = 1e-2
    time_schedule = LinearTimeSchedule()
    num_sampling_steps = 10
    times = time_schedule(t0_sampling, t1, num_sampling_steps)
    sample_size = 1000
    conds = None
    sampler = ScheduledEulerSampler(times=times)

    # 9. guidance
    a = 1.0
    mask = jnp.array([True, False])
    y = jnp.array([a])
    guidance = ManifoldGuidance(mask)
