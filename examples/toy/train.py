import argparse

import jax.numpy as jnp
import jax.random as jr
from configs import get_config
from reference import get_joint

from confusion.checkpointing import Checkpointer
from confusion.training import train
from confusion.utils import normalize


def main(args):
    # get config
    config = get_config(args)
    seed = config.seed
    num_samples = config.num_samples
    num_steps = config.num_steps
    batch_size = config.batch_size
    model = config.model
    opt = config.opt
    print_every = config.print_every
    saving_path = config.saving_path
    max_save_to_keep = config.max_save_to_keep
    save_every = config.save_every
    train_key = config.train_key
    t0 = config.t0
    t1 = config.t1

    # generate samples
    key = jr.PRNGKey(seed)
    A, B = get_joint(num_samples, key)
    samples = jnp.concatenate([A, B], axis=1)
    samples, _, _ = normalize(samples)

    # get checkpointer for new checkpoints
    ckpter = Checkpointer(
        saving_path,
        max_save_to_keep,
        save_every,
        erase=True,
    )

    # training
    train(
        model,
        opt,
        samples,
        num_steps,
        batch_size,
        print_every,
        ckpter,
        train_key,
        None,
        t0=t0,
        t1=t1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training with given configuration."
    )
    parser.add_argument(
        "--config",
        choices=[
            "cve",
            "cvp",
            "ve",
            "vp",
        ],
        required=True,
    )
    args = parser.parse_args()

    main(args)
