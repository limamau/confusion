import os

import jax.numpy as jnp
import jax.random as jr
import ml_collections
import optax

from confusion.diffusion import VarianceExploding
from confusion.guidance import MomentMatchingGuidance
from confusion.networks import MultiLayerPerceptron
from confusion.sampling import ODESampler


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = "ve_mlp"

    # dataset
    config.num_samples = 10_000

    # keys
    config.seed = 5678
    key = jr.PRNGKey(config.seed)
    config.net_key, config.train_key, config.sample_key = jr.split(key, 3)

    # network
    config.t1 = 3.0
    config.num_variables = 3
    config.hidden_size = 256
    config.is_conditional = False
    config.network = MultiLayerPerceptron(
        config.num_variables,
        config.hidden_size,
        config.t1,
        key=config.net_key,
        is_conditional=config.is_conditional,
    )

    # diffusion model
    config.t0 = 0.1
    config.sigma_min = 0.1
    config.sigma_max = 0.12
    config.weight_fn = lambda t: config.sigma_min * jnp.pow(
        (config.sigma_max / config.sigma_min), 2 * t
    )  # weight is taken to be equal to sigma(t)^2
    config.is_approximate = False
    config.model = VarianceExploding(
        config.network,
        config.weight_fn,
        config.t0,
        config.t1,
        config.sigma_min,
        config.sigma_max,
        is_approximate=config.is_approximate,
    )

    # optimization
    config.num_steps = 10_000
    config.lr = 1e-3
    config.batch_size = 16
    config.opt = optax.adam(config.lr)

    # logging and checkpointing
    config.print_every = 1000
    config.max_save_to_keep = 1
    config.save_every = 5_000
    config.saving_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../checkpoints/{config.experiment_name}",
    )

    # sampling
    config.dt0 = 0.01
    config.sampler = ODESampler(config.dt0, config.t1)
    config.sample_size = 1000
    config.conds = None
    config.do_B = 1.0
    config.const_matrix = jnp.array([[0.0, config.do_B, 0.0]])
    config.y = jnp.array([config.do_B])
    config.guidance = MomentMatchingGuidance(
        config.const_matrix,
        config.y,
    )

    return config
