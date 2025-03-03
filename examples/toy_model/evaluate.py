import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
from configs import get_config
from experiment import get_joint

from confusion.checkpointing import Checkpointer
from confusion.utils import normalize


def main(args):
    # get config
    config = get_config(args)
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

    # generate samples
    ref_A, ref_B, ref_C = get_joint(sample_size)
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

    # sampling
    gen_samples = sampler.sample(
        model,
        ref_samples.shape[1:],
        conds,
        sample_key,
        ref_samples_mean,
        ref_samples_std,
        sample_size,
    )
    gen_A, gen_B, gen_C = jnp.split(gen_samples, 3, axis=1)

    # comparisons without intervention
    # for ref, gen, label in (
    #     (ref_A, gen_A, "A"),
    #     (ref_B, gen_B, "B"),
    #     (ref_C, gen_C, "C"),
    # ):
    #     plt.hist(ref.flatten(), bins=20, alpha=0.5, label="ref")
    #     plt.hist(gen.flatten(), bins=20, alpha=0.5, label="gen")
    #     plt.title(f"No Intervention ({label})")
    #     plt.legend()
    #     plt.show()

    # # ref histogram
    # plt.figure(figsize=(6, 4))
    # plt.hist(ref_A.flatten(), bins=20, alpha=0.5, label="A")
    # plt.hist(ref_B.flatten(), bins=20, alpha=0.5, label="B")
    # plt.hist(ref_C.flatten(), bins=20, alpha=0.5, label="C")
    # plt.title("Reference distributions")
    # plt.legend()
    # plt.show()

    # # gen histogram
    # plt.figure(figsize=(6, 4))
    # plt.hist(gen_A.flatten(), bins=20, alpha=0.5, label="A")
    # plt.hist(gen_B.flatten(), bins=20, alpha=0.5, label="B")
    # plt.hist(gen_C.flatten(), bins=20, alpha=0.5, label="C")
    # plt.title("Generated distributions")
    # plt.legend()
    # plt.show()

    # # show B as a function of A
    # plt.scatter(
    #     gen_A.flatten(),
    #     gen_B.flatten(),
    #     alpha=0.1,
    #     label="gen",
    # )
    # plt.scatter(
    #     jnp.sort(ref_A.flatten()),
    #     jnp.sort(ref_B.flatten()),
    #     alpha=0.1,
    #     label="ref",
    # )
    # plt.title("p(B | A)")
    # plt.xlabel("A")
    # plt.ylabel("B")
    # plt.legend()
    # plt.show()

    # no intervention - reference
    plt.figure(figsize=(3, 2), dpi=200)
    ref_A, ref_B, ref_C = get_joint(sample_size)
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
    ref_A, ref_B, ref_C = get_joint(sample_size, do_B=do_B)
    plt.hist(ref_A.flatten(), bins=20, alpha=0.5, label="A")
    plt.hist(ref_B.flatten(), bins=20, alpha=0.5, label="B")
    plt.hist(ref_C.flatten(), bins=20, alpha=0.5, label="C")
    plt.title("Do(B={}) - reference".format(do_B))
    plt.xlim(-5, 5)
    plt.ylim(0, 200)
    plt.legend()
    plt.show()

    # no intervention - diffusion experiment
    plt.figure(figsize=(3, 2), dpi=200)
    plt.hist(gen_A.flatten(), bins=20, alpha=0.5, label="A")
    plt.hist(gen_B.flatten(), bins=20, alpha=0.5, label="B")
    plt.hist(gen_C.flatten(), bins=20, alpha=0.5, label="C")
    plt.title("Diffusion generation")
    plt.xlim(-5, 5)
    plt.ylim(0, 200)
    plt.legend()
    plt.show()

    # do(B) intervention
    # TO BE DONE


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
