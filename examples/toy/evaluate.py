import argparse
import time

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from configs import get_config
from reference import get_joint

from confusion.checkpointing import Checkpointer
from confusion.utils import normalize

FIGSIZE = (7, 3)
BINS = 20
ALPHA = 0.5
XLIM = (-5, 5)
YLIM = (0, 200)


def print_mean_and_variance(samples_A, samples_B):
    mean_A = jnp.mean(samples_A)
    std_A = jnp.std(samples_A)
    mean_B = jnp.mean(samples_B)
    std_B = jnp.std(samples_B)
    print(f"A: mean={mean_A:.2f}, std={std_A:.2f}")
    print(f"B: mean={mean_B:.2f}, std={std_B:.2f}")
    print()


def main(args):
    # get config
    config = get_config(args)
    seed = config.seed
    model = config.model
    opt = config.opt
    saving_path = config.saving_path
    max_save_to_keep = config.max_save_to_keep
    save_every = config.save_every
    sample_size = config.sample_size
    sample_key = config.sample_key
    conds = config.conds
    num_variables = config.num_variables
    do_A = config.do_A
    sampler = config.sampler
    guidance = config.guidance

    # generate samples
    key = jr.PRNGKey(seed)
    ref_A, ref_B = get_joint(sample_size, key)
    ref_samples = jnp.concatenate([ref_A, ref_B], axis=1)
    ref_samples, ref_samples_mean, ref_samples_std = normalize(ref_samples)

    # get checkpointer to restore
    ckpter = Checkpointer(
        saving_path,
        max_save_to_keep,
        save_every,
    )

    # restore
    model, _ = ckpter.restore(model, opt)

    # no intervention - reference
    ref_A, ref_B = get_joint(sample_size, key)
    plt.figure(figsize=FIGSIZE)
    plt.subplot(1, 2, 1)
    title = "No intervention - reference"
    print(title)
    print_mean_and_variance(ref_A, ref_B)
    plt.hist(ref_A.flatten(), bins=BINS, alpha=ALPHA, label="A")
    plt.hist(ref_B.flatten(), bins=BINS, alpha=ALPHA, label="B")
    plt.title(title)
    plt.xlim(*XLIM)
    plt.ylim(*YLIM)
    plt.legend()

    # no intervention - diffusion model
    plt.subplot(1, 2, 2)
    start_time = time.time()
    gen_samples = sampler.sample(
        model,
        ref_samples.shape[1:],
        conds,
        sample_key,
        ref_samples_mean,
        ref_samples_std,
        sample_size,
    )
    end_time = time.time()
    gen_A, gen_B = jnp.split(gen_samples, num_variables, axis=1)
    title = "No intervention - diffusion model"
    print(title)
    print("Sampling time: {:.2f} seconds".format(end_time - start_time))
    print_mean_and_variance(gen_A, gen_B)
    plt.hist(gen_A.flatten(), bins=BINS, alpha=ALPHA, label="A")
    plt.hist(gen_B.flatten(), bins=BINS, alpha=ALPHA, label="B")
    plt.title(title)
    plt.xlim(*XLIM)
    plt.ylim(*YLIM)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # do(B) intervention - reference
    plt.figure(figsize=FIGSIZE)
    plt.subplot(1, 2, 1)
    ref_A, ref_B = get_joint(sample_size, key, do_A=do_A)
    print("Do(B={}) - reference".format(do_A))
    print_mean_and_variance(ref_A, ref_B)
    plt.hist(ref_A.flatten(), bins=BINS, alpha=ALPHA, label="A")
    plt.hist(ref_B.flatten(), bins=BINS, alpha=ALPHA, label="B")
    plt.title("Do(B={}) - reference".format(do_A))
    plt.xlim(*XLIM)
    plt.ylim(*YLIM)
    plt.legend()

    # do(B) intervention - diffusion model
    plt.subplot(1, 2, 2)
    start_time = time.time()
    gen_samples = sampler.sample(
        model,
        ref_samples.shape[1:],
        conds,
        sample_key,
        ref_samples_mean,
        ref_samples_std,
        sample_size,
        guidance=guidance,
    )
    end_time = time.time()
    gen_A, gen_B = jnp.split(gen_samples, num_variables, axis=1)
    print("Do(B={}) - diffusion model".format(do_A))
    print("Sampling time: {:.2f} seconds".format(end_time - start_time))
    print_mean_and_variance(gen_A, gen_B)
    plt.hist(gen_A.flatten(), bins=BINS, alpha=ALPHA, label="A")
    plt.hist(gen_B.flatten(), bins=BINS, alpha=ALPHA, label="B")
    plt.title("Do(B={}) - diffusion model".format(do_A))
    plt.xlim(*XLIM)
    plt.ylim(*YLIM)
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
            "vp",
        ],
        required=True,
    )
    args = parser.parse_args()

    main(args)
