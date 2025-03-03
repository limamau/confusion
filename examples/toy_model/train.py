import argparse

import jax.numpy as jnp
from configs import get_config
from experiment import get_joint

from confusion.checkpointing import Checkpointer
from confusion.training import train
from confusion.utils import normalize


def main(args):
    # get config
    config = get_config(args)
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

    # generate samples
    A, B, C = get_joint(num_samples)
    samples = jnp.concatenate([A, B, C], axis=1)
    print("joint shape:", samples.shape)
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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training with given configuration."
    )
    parser.add_argument(
        "--config",
        choices=[
            "ve",
            "vp",
        ],
        required=True,
    )
    args = parser.parse_args()

    main(args)
