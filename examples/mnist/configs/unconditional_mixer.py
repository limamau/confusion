import ml_collections, optax, os
import jax.numpy as jnp
import jax.random as jr

from confusion.networks import Mixer2d


def get_config(imgs_shape):
    config = ml_collections.ConfigDict()
    config.experiment_name = "unconditional_mixer"
    
    # keys
    config.seed = 5678
    key = jr.PRNGKey(config.seed)
    config.net_key, config.train_key, config.sample_key = jr.split(key, 3)

    # denoiser network
    config.patch_size = 4
    config.hidden_size = 64
    config.mix_patch_size = 512
    config.mix_hidden_size = 512
    config.num_blocks = 4
    config.t1 = 10.0
    config.is_conditional = False
    config.network = Mixer2d(
        imgs_shape,
        config.patch_size,
        config.hidden_size,
        config.mix_patch_size,
        config.mix_hidden_size,
        config.num_blocks,
        config.t1,
        key=config.net_key,
        is_conditional=config.is_conditional,
    )
    
    # noise parameterisations
    config.int_beta = lambda t: t
    # (just chosen to upweight the region near t=0)
    config.weight = lambda t: 1 - jnp.exp(
        -config.int_beta(t)
    )
    
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
        f"../checkpoints/{config.experiment_name}"
    )
    
    # sampling
    config.dt0 = 0.1
    config.sample_size = 10
    config.conds = None
    
    return config
