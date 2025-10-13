import jax.random as jr
from config import Config
from dynamics import generate_dataset, split_in_conds

from confusion.checkpointing import Checkpointer
from confusion.training import train

# from confusion.utils import normalize


def main():
    # get config
    config = Config()
    seed = config.seed
    # num_samples = config.num_samples
    num_train_steps = config.num_steps
    train_batch_size = config.train_batch_size
    model = config.model
    opt = config.opt
    print_loss_every = config.print_loss_every
    eval_batch_size = config.eval_batch_size
    eval_every = config.eval_every
    saving_path = config.saving_path
    max_save_to_keep = config.max_save_to_keep
    save_every = config.save_every
    key = config.train_key
    t0 = config.t0_training
    t1 = config.t1
    time_schedule = config.time_schedule
    # sigma_data = config.sigma_data

    # generate training and validation samples
    key = jr.PRNGKey(seed)
    dataset_key, train_key = jr.split(key)
    train_data, eval_data, _ = generate_dataset(dataset_key)
    train_data, train_conds = split_in_conds(train_data)
    eval_data, eval_conds = split_in_conds(eval_data)

    # get checkpointer for new checkpoints
    ckpter = Checkpointer(
        saving_path,
        max_save_to_keep,
        save_every,
        erase=True,
        saving_criteria="best",
    )

    # training
    train(
        model,
        opt,
        train_data,
        eval_data,
        num_train_steps,
        train_batch_size,
        print_loss_every,
        eval_batch_size,
        eval_every,
        train_key,
        train_conds=train_conds,
        eval_conds=eval_conds,
        t0=t0,
        t1=t1,
        ckpter=ckpter,
        time_schedule=time_schedule,
    )


if __name__ == "__main__":
    main()
