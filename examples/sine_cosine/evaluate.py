import argparse
import time

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from configs import get_config
from jaxtyping import Array
from reference import get_joint

from confusion.checkpointing import Checkpointer
from confusion.utils import denormalize, normalize


def print_mean_and_variance(samples: Array) -> None:
    sin_samples, cos_samples = jnp.split(samples, 2, axis=1)
    mean_sin = jnp.mean(sin_samples)
    std_sin = jnp.std(sin_samples)
    mean_cos = jnp.mean(cos_samples)
    std_cos = jnp.std(cos_samples)
    print(f"sin: mean={mean_sin:.2f}, std={std_sin:.2f}")
    print(f"cos: mean={mean_cos:.2f}, std={std_cos:.2f}")
    print()


def main(args):
    # get config
    config = get_config(args)
    seq_len = config.seq_len
    seed = config.seed
    model = config.model
    opt = config.opt
    sigma_data = config.sigma_data
    saving_path = config.saving_path
    max_save_to_keep = config.max_save_to_keep
    save_every = config.save_every
    sample_size = config.sample_size
    evaluate_key = config.evaluate_key
    pre_conds = config.conds
    sampler = config.sampler

    # generate samples
    key = jr.PRNGKey(seed)
    ref_samples = get_joint(sample_size, seq_len, key)
    _, ref_samples_mean, ref_samples_std = normalize(
        ref_samples, imposed_std=sigma_data
    )
    print("ref mean:", ref_samples_mean)
    print("ref std:", ref_samples_std)

    # get checkpointer to restore
    ckpter = Checkpointer(
        saving_path,
        max_save_to_keep,
        save_every,
        saving_criteria="best",
    )

    # restore
    model, _ = ckpter.restore(model, opt)

    # reference
    plt.figure()
    plt.subplot(1, 2, 1)
    title = "Reference"
    print(title)
    print_mean_and_variance(ref_samples)
    ref_mean = jnp.mean(ref_samples, axis=0)
    upper = jnp.percentile(ref_samples, 95, axis=0)
    lower = jnp.percentile(ref_samples, 5, axis=0)
    plt.plot(ref_mean[0], label="sin")
    plt.plot(ref_mean[1], label="cos")
    plt.fill_between(range(seq_len), lower[0], upper[0], alpha=0.2)
    plt.fill_between(range(seq_len), lower[1], upper[1], alpha=0.2)
    plt.title(title)
    plt.legend()

    # Diffusion model
    plt.subplot(1, 2, 2)
    start_time = time.time()
    gen_samples = sampler.sample(
        model,
        ref_samples.shape[1:],
        pre_conds,
        None,
        evaluate_key,
        sample_size,
    )
    gen_samples = denormalize(
        gen_samples, ref_samples_mean, ref_samples_std, imposed_std=sigma_data
    )
    end_time = time.time()
    title = "Diffusion model"
    print(title)
    print("Sampling time: {:.2f} seconds".format(end_time - start_time))
    print_mean_and_variance(gen_samples)
    gen_mean = jnp.mean(gen_samples, axis=0)
    upper = jnp.percentile(gen_samples, 95, axis=0)
    lower = jnp.percentile(gen_samples, 5, axis=0)
    plt.plot(gen_mean[0], label="sin")
    plt.plot(gen_mean[1], label="cos")
    plt.fill_between(range(seq_len), lower[0], upper[0], alpha=0.2)
    plt.fill_between(range(seq_len), lower[1], upper[1], alpha=0.2)
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation with given configuration."
    )
    parser.add_argument(
        "--config",
        choices=[
            "ve",
        ],
        required=True,
    )
    args = parser.parse_args()

    main(args)
