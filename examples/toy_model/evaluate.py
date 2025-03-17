import argparse
import os
import time

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from configs import get_config
from experiment import get_joint, print_mean_and_variance

from confusion.checkpointing import Checkpointer
from confusion.utils import get_and_create_figs_dir, normalize

FIGSIZE = (3, 2)
DPI = 200
BINS = 20
ALPHA = 0.5
XLIM = (-6, 6)
YLIM = (0, 200)
FIGS_DIR = get_and_create_figs_dir(__file__)


def get_file_name(figs_dir: str, title: str) -> str:
    return os.path.join(
        figs_dir, title.replace(" ", "").replace(",", "-").replace(".", "-") + ".png"
    )


def main(args):
    # get config
    config = get_config(args)
    name = config.name
    seed = config.seed
    model = config.model
    opt = config.opt
    saving_path = config.saving_path
    max_save_to_keep = config.max_save_to_keep
    save_every = config.save_every
    sample_size = config.sample_size
    sample_key = config.sample_key
    conds = config.conds
    std_sampler = config.std_sampler
    edm_sampler = config.edm_sampler
    do_B = config.do_B
    moment_matching_guidance = config.moment_matching_guidance

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
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ref_A, ref_B, ref_C = get_joint(sample_size, key)
    title = "No intervention - reference"
    print(title)
    print_mean_and_variance(ref_A, ref_B, ref_C)
    plt.hist(ref_A.flatten(), bins=BINS, alpha=ALPHA, label="A")
    plt.hist(ref_B.flatten(), bins=BINS, alpha=ALPHA, label="B")
    plt.hist(ref_C.flatten(), bins=BINS, alpha=ALPHA, label="C")
    plt.title(title)
    plt.xlim(*XLIM)
    plt.ylim(*YLIM)
    plt.legend()
    plt.show()
    plt.close()
    file_name = get_file_name(FIGS_DIR, title)
    fig.savefig(file_name)

    # do(B) intervention - reference
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ref_A, ref_B, ref_C = get_joint(sample_size, key, do_B=do_B)
    title = "Do(B={}) - reference".format(do_B)
    print(title)
    print_mean_and_variance(ref_A, ref_B, ref_C)
    plt.hist(ref_A.flatten(), bins=BINS, alpha=ALPHA, label="A")
    plt.hist(ref_B.flatten(), bins=BINS, alpha=ALPHA, label="B")
    plt.hist(ref_C.flatten(), bins=BINS, alpha=ALPHA, label="C")
    plt.title(title)
    plt.xlim(*XLIM)
    plt.ylim(*YLIM)
    plt.legend()
    plt.show()
    plt.close()
    file_name = get_file_name(FIGS_DIR, title)
    fig.savefig(file_name)

    # no intervention - diffusion experiment
    # std sampling with no guidance
    start_time = time.time()
    gen_samples = std_sampler.sample(
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
    title = "No intervention - {}, {}".format(name.upper(), "std. sampler")
    print(title)
    print("Sampling time: {:.2f} seconds".format(end_time - start_time))
    print_mean_and_variance(gen_A, gen_B, gen_C)
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.hist(gen_A.flatten(), bins=BINS, alpha=ALPHA, label="A")
    plt.hist(gen_B.flatten(), bins=BINS, alpha=ALPHA, label="B")
    plt.hist(gen_C.flatten(), bins=BINS, alpha=ALPHA, label="C")
    plt.title(title)
    plt.xlim(*XLIM)
    plt.ylim(*YLIM)
    plt.legend()
    plt.show()
    plt.close()
    file_name = get_file_name(FIGS_DIR, title)
    fig.savefig(file_name)

    # do(B) intervention - diffusion experiment
    # std sampling with guidance
    start_time = time.time()
    gen_samples = std_sampler.sample(
        model,
        ref_samples.shape[1:],
        conds,
        sample_key,
        ref_samples_mean,
        ref_samples_std,
        sample_size,
        guidance=moment_matching_guidance,
    )
    end_time = time.time()
    gen_A, gen_B, gen_C = jnp.split(gen_samples, 3, axis=1)
    title = "Do(B={}) - {}, {}".format(do_B, name.upper(), "std. sampler")
    print(title)
    print("Sampling time: {:.2f} seconds".format(end_time - start_time))
    print_mean_and_variance(gen_A, gen_B, gen_C)
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.hist(gen_A.flatten(), bins=BINS, alpha=ALPHA, label="A")
    plt.hist(gen_B.flatten(), bins=BINS, alpha=ALPHA, label="B")
    plt.hist(gen_C.flatten(), bins=BINS, alpha=ALPHA, label="C")
    plt.title(title)
    plt.xlim(*XLIM)
    plt.ylim(*YLIM)
    plt.legend()
    plt.show()
    plt.close()
    file_name = get_file_name(FIGS_DIR, title)
    fig.savefig(file_name)

    # do(B) intervention - diffusion experiment
    # edm sampling with guidance
    start_time = time.time()
    gen_samples = edm_sampler.sample(
        model,
        ref_samples.shape[1:],
        conds,
        sample_key,
        ref_samples_mean,
        ref_samples_std,
        sample_size,
        guidance=moment_matching_guidance,
    )
    end_time = time.time()
    gen_A, gen_B, gen_C = jnp.split(gen_samples, 3, axis=1)
    title = "Do(B={}) - {}, {}".format(do_B, name.upper(), "EDM sampler")
    print(title)
    print("Sampling time: {:.2f} seconds".format(end_time - start_time))
    print_mean_and_variance(gen_A, gen_B, gen_C)
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.hist(gen_A.flatten(), bins=BINS, alpha=ALPHA, label="A")
    plt.hist(gen_B.flatten(), bins=BINS, alpha=ALPHA, label="B")
    plt.hist(gen_C.flatten(), bins=BINS, alpha=ALPHA, label="C")
    plt.title(title)
    plt.xlim(*XLIM)
    plt.ylim(*YLIM)
    plt.legend()
    plt.show()
    plt.close()
    file_name = get_file_name(FIGS_DIR, title)
    fig.savefig(file_name)


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
