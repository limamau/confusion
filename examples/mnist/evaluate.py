import argparse, einops, jax, os
import equinox as eqx
import functools as ft
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from confusion.checkpointing import Checkpointer
from confusion.sampling import single_ode_sample_fn
from confusion.utils import normalize

from configs import get_config
from utils import load_mnist


def main(args):
    # dataset download/load and normalization
    images, labels = load_mnist()
    images, images_mean, images_std = normalize(images)
    _, labels_mean, labels_std = normalize(labels)
    images_shape = images.shape[1:]

    # get config
    config = get_config(args, images_shape)
    network = config.network
    is_conditional = config.is_conditional
    int_beta = config.int_beta
    t1 = config.t1
    opt = config.opt
    saving_path = config.saving_path
    max_save_to_keep = config.max_save_to_keep
    save_every = config.save_every
    dt0 = config.dt0
    sample_size = config.sample_size
    conds = config.conds
    sample_key = config.sample_key
    experiment_name = config.experiment_name

    # get checkpointer to restore
    ckpter = Checkpointer(
        saving_path,
        max_save_to_keep,
        save_every,
    )

    # restore
    opt_state = opt.init(eqx.filter(network, eqx.is_inexact_array))
    network, _ = ckpter.restore(network, opt_state)

    # sampling
    if is_conditional:
        conds, _, _ = normalize(conds, labels_mean, labels_std)
    sample_key = jr.split(sample_key, sample_size**2)
    sample_fn = ft.partial(
        single_ode_sample_fn, network, int_beta, images_shape, dt0, t1,
    )
    sample = jax.vmap(sample_fn)(conds, sample_key)
    sample = images_mean + images_std * sample
    images_max = jnp.max(images)
    images_min = jnp.min(images)
    sample = jnp.clip(sample, images_min, images_max)
    sample = einops.rearrange(
        sample, "(n1 n2) 1 h w -> (n1 h) (n2 w)", n1=sample_size, n2=sample_size,
    )
    plt.imshow(sample, cmap="Greys")
    plt.axis("off")
    plt.tight_layout()
    figs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "figs",
    )
    os.makedirs(figs_dir, exist_ok=True)
    plt.savefig(os.path.join(figs_dir, f"{experiment_name}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation with given configuration."
    )
    parser.add_argument(
        "--config",
        choices=[
            "conditional_mixer",
            "conditional_unet",
            "unconditional_mixer",
            "unconditional_unet",
        ],
        required=True,
    )
    args = parser.parse_args()

    main(args)
