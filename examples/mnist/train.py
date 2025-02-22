import argparse

from confusion.checkpointing import Checkpointer
from confusion.training import train
from confusion.utils import normalize

from configs import get_config
from utils import load_mnist


def main(args):
    # dataset download/load and normalization
    images, labels = load_mnist()
    images, _, _ = normalize(images)
    labels, _, _ = normalize(labels)
    imgs_shape = images.shape[1:]

    # get config
    config = get_config(args, imgs_shape)
    num_steps = config.num_steps
    batch_size = config.batch_size
    network = config.network
    is_conditional = config.is_conditional
    t1 = config.t1
    weight = config.weight
    int_beta = config.int_beta
    opt = config.opt
    print_every = config.print_every
    saving_path = config.saving_path
    max_save_to_keep = config.max_save_to_keep
    save_every = config.save_every
    train_key = config.train_key

    # use labels as condition or not
    if not is_conditional:
        labels = None

    # get checkpointer for new checkpoints
    ckpter = Checkpointer(
        saving_path,
        max_save_to_keep,
        save_every,
        erase=True,
    )

    # training
    train(
        num_steps,
        images,
        batch_size,
        network,
        weight,
        int_beta,
        t1,
        opt,
        print_every,
        ckpter,
        train_key,
        labels,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training with given configuration."
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
