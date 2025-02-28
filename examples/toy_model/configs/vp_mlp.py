import os

import jax.numpy as jnp
import jax.random as jr
import ml_collections
import optax

from confusion.diffusion import VPDiffusionModel
from confusion.networks import MLP
from confusion.sampling import ODESampler


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = "vp_mlp"

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
    config.network = MLP(
        config.num_variables,
        config.hidden_size,
        config.t1,
        key=config.net_key,
        is_conditional=config.is_conditional,
    )

    # noise parameterisations
    config.int_beta_fn = lambda t: t
    config.weight_fn = lambda t: 1 - jnp.exp(-config.int_beta_fn(t))

    # diffusion model
    config.model = VPDiffusionModel(
        config.network, config.int_beta_fn, config.weight_fn
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
    config.sample_size = 1000
    config.conds = None
    config.do_B = 3
    config.sampler = ODESampler(config.dt0, config.t1)

    return config
