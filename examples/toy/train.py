import argparse

import jax.numpy as jnp
import jax.random as jr
from configs import get_config
from reference import get_joint

from confusion.checkpointing import Checkpointer
from confusion.training import train


def main(args):
    # get config
    config = get_config(args)
    seed = config.seed
    num_samples = config.num_samples
    num_train_steps = config.num_steps
    train_batch_size = config.train_batch_size
    model = config.model
    opt = config.opt
    loss = config.loss
    print_loss_every = config.print_loss_every
    eval_batch_size = config.eval_batch_size
    eval_every = config.eval_every
    saving_path = config.saving_path
    max_save_to_keep = config.max_save_to_keep
    save_every = config.save_every
    key = config.train_key
    t0 = config.t0
    t1 = config.t1

    # generate training and validation samples
    key = jr.PRNGKey(seed)
    keys = jr.split(key, 3)
    A, B = get_joint(num_samples, keys[0])
    train_data = jnp.concatenate([A, B], axis=1)
    A, B = get_joint(num_samples, keys[1])
    eval_data = jnp.concatenate([A, B], axis=1)

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
        loss,
        train_data,
        eval_data,
        num_train_steps,
        train_batch_size,
        print_loss_every,
        eval_batch_size,
        eval_every,
        ckpter,
        keys[2],
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
            "edm",
            "ve",
            "vp",
        ],
        required=True,
    )
    args = parser.parse_args()

    main(args)
