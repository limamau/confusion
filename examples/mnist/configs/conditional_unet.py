import os

import jax.numpy as jnp
import jax.random as jr
import ml_collections
import optax

from confusion.networks.images import UNet


def get_config(imgs_shape):
    config = ml_collections.ConfigDict()
    config.experiment_name = "conditional_unet"

    # keys
    config.seed = 5678
    key = jr.PRNGKey(config.seed)
    config.net_key, config.train_key, config.sample_key = jr.split(key, 3)

    # denoiser network
    config.is_biggan = True
    config.dim_mults = [2, 4, 8]
    config.hidden_size = 16
    config.heads = 1
    config.dim_head = 8
    config.dropout_rate = 0.2
    config.num_res_blocks = 3
    config.attn_resolutions = [7]
    config.is_conditional = True
    config.network = UNet(
        imgs_shape,
        config.is_biggan,
        config.dim_mults,
        config.hidden_size,
        config.heads,
        config.dim_head,
        config.dropout_rate,
        config.num_res_blocks,
        config.attn_resolutions,
        key=config.net_key,
        is_conditional=config.is_conditional,
    )

    # noise parameterisations
    config.t1 = 10.0
    config.int_beta = lambda t: t
    # (just chosen to upweight the region near t=0)
    config.weight = lambda t: 1 - jnp.exp(-config.int_beta(t))

    # optimization
    config.num_steps = 1_000_000
    config.lr = 3e-4
    config.batch_size = 256
    config.opt = optax.adabelief(config.lr)

    # logging and checkpointing
    config.print_every = 10_000
    config.max_save_to_keep = 1
    config.save_every = 100_000
    config.saving_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../checkpoints/{config.experiment_name}",
    )

    # sampling
    config.dt0 = 0.1
    config.sample_size = 10
    config.conds = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10)

    return config
