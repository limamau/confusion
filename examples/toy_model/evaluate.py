import argparse
import time

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from configs import get_config
from experiment import get_joint, print_mean_and_variance

from confusion.checkpointing import Checkpointer
from confusion.utils import normalize


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
    sampler = config.sampler
    do_B = config.do_B
    guidance = config.guidance

    # generate samples
    key = jr.PRNGKey(seed)
    ref_A, ref_B, ref_C = get_joint(sample_size, key)
    ref_samples = jnp.concatenate([ref_A, ref_B, ref_C], axis=1)
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
    plt.figure(figsize=(3, 2), dpi=200)
    ref_A, ref_B, ref_C = get_joint(sample_size, key)
    print("No intervention - reference")
    print_mean_and_variance(ref_A, ref_B, ref_C)
    plt.hist(ref_A.flatten(), bins=20, alpha=0.5, label="A")
    plt.hist(ref_B.flatten(), bins=20, alpha=0.5, label="B")
    plt.hist(ref_C.flatten(), bins=20, alpha=0.5, label="C")
    plt.title("No Intervention - reference")
    plt.xlim(-5, 5)
    plt.ylim(0, 200)
    plt.legend()
    plt.show()

    # do(B) intervention - reference
    plt.figure(figsize=(3, 2), dpi=200)
    ref_A, ref_B, ref_C = get_joint(sample_size, key, do_B=do_B)
    print("Do(B={}) - reference".format(do_B))
    print_mean_and_variance(ref_A, ref_B, ref_C)
    plt.hist(ref_A.flatten(), bins=20, alpha=0.5, label="A")
    plt.hist(ref_B.flatten(), bins=20, alpha=0.5, label="B")
    plt.hist(ref_C.flatten(), bins=20, alpha=0.5, label="C")
    plt.title("Do(B={}) - reference".format(do_B))
    plt.xlim(-5, 5)
    plt.ylim(0, 200)
    plt.legend()
    plt.show()

    # no intervention - diffusion experiment
    # (sampling with no guidance)
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
    gen_A, gen_B, gen_C = jnp.split(gen_samples, 3, axis=1)
    print("No intervention - diffusion model")
    print("Sampling time: {:.2f} seconds".format(end_time - start_time))
    print_mean_and_variance(gen_A, gen_B, gen_C)
    plt.figure(figsize=(3, 2), dpi=200)
    plt.hist(gen_A.flatten(), bins=20, alpha=0.5, label="A")
    plt.hist(gen_B.flatten(), bins=20, alpha=0.5, label="B")
    plt.hist(gen_C.flatten(), bins=20, alpha=0.5, label="C")
    plt.title("No Intervention - diffusion model")
    plt.xlim(-5, 5)
    plt.ylim(0, 200)
    plt.legend()
    plt.show()

    # do(B) intervention - diffusion experiment
    # (sampling with guidance)
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
    gen_A, gen_B, gen_C = jnp.split(gen_samples, 3, axis=1)
    print("Do(B={}) - diffusion model".format(do_B))
    print("Sampling time: {:.2f} seconds".format(end_time - start_time))
    print_mean_and_variance(gen_A, gen_B, gen_C)
    plt.figure(figsize=(3, 2), dpi=200)
    plt.hist(gen_A.flatten(), bins=20, alpha=0.5, label="A")
    plt.hist(gen_B.flatten(), bins=20, alpha=0.5, label="B")
    plt.hist(gen_C.flatten(), bins=20, alpha=0.5, label="C")
    plt.title("Do(B={}) - diffusion model".format(do_B))
    plt.xlim(-5, 5)
    plt.ylim(0, 200)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation with given configuration."
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
