import functools as ft
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from config import Config
from dynamics import generate_dataset, generate_ts, split_in_conds
from jaxtyping import Array

from confusion.checkpointing import Checkpointer


def calculate_relative_error(true_ys: Array, pred_ys: Array):
    """
    Calculates the mean relative error over time for a batch of trajectories.
    The error at each time step is ||true - pred|| / (||true|| + ||pred|| + epsilon).
    """
    # L2 norm of the error at each time step for each trajectory
    norm_over_points_fn = ft.partial(jnp.linalg.norm, axis=0)
    error_norm = jax.vmap(norm_over_points_fn)(true_ys - pred_ys)

    # L2 norm of the true state at each time step
    true_norm = jax.vmap(norm_over_points_fn)(true_ys)

    # L2 norm of the predicted state at each time step
    pred_norm = jax.vmap(norm_over_points_fn)(pred_ys)

    # add a small epsilon to the denominator to avoid division by zero
    epsilon = 1e-8
    relative_error = error_norm / (true_norm + pred_norm + epsilon)

    # average the relative error across all trajectories in the batch
    avg_rel_error = jnp.exp(jnp.mean(jnp.log(relative_error), axis=0))
    return avg_rel_error


def plot_relative_error(ts: Array, avg_rel_error: Array):
    """Plots the relative error over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(ts, avg_rel_error)
    plt.ylim(1e-3, 1)
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Error")
    plt.yscale("log")
    plt.show()


def main():
    # get config
    config = Config()
    seed = config.seed
    model = config.model
    opt = config.opt
    saving_path = config.saving_path
    max_save_to_keep = config.max_save_to_keep
    save_every = config.save_every
    evaluate_key = config.evaluate_key
    sampler = config.sampler

    # generate training and validation samples
    key = jr.PRNGKey(seed)
    dataset_key, _ = jr.split(key)
    _, _, test_data = generate_dataset(dataset_key)
    test_data, test_conds = split_in_conds(test_data)

    # get checkpointer to restore
    ckpter = Checkpointer(
        saving_path,
        max_save_to_keep,
        save_every,
        saving_criteria="best",
    )

    # restore
    model, _ = ckpter.restore(model, opt)

    # generate samples
    print("generating samples...")
    start_time = time.time()
    samples = jax.block_until_ready(
        sampler.sample(
            model,
            test_data.shape[1:],
            test_conds,
            None,
            evaluate_key,
            test_data.shape[0],
        )
    )
    end_time = time.time()
    print(f"generation time: {round(end_time - start_time)}s")
    print("samples shape:", samples.shape)

    avg_rel_errors = calculate_relative_error(test_data, samples)
    ts = generate_ts()
    print("ts shape:", ts.shape)
    print("relative error shape:", avg_rel_errors.shape)
    plot_relative_error(ts, avg_rel_errors)


if __name__ == "__main__":
    main()
