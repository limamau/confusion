import argparse
import time

import jax.numpy as jnp
import jax.random as jr
from configs import get_config
from reference import get_joint, plot_samples

from confusion.checkpointing import Checkpointer
from confusion.utils import get_and_create_figs_dir, normalize


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
    do_B = config.do_B
    # sampler options
    em_sampler = config.em_sampler
    edm_sampler = config.edm_sampler
    # guidance options
    guidance_free = config.guidance_free
    moment_matching_guidance = config.moment_matching_guidance
    manifold_guidance = config.manifold_guidance
    figs_dir = get_and_create_figs_dir(__file__, name)

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

    # diffusion experiments
    for title, do_B, guidance, sampler in [
        (
            "No intervention - {}, EM".format(name.upper()),
            None,
            guidance_free,
            em_sampler,
        ),
        (
            "Do(B={}) - {}, MM-G, EM".format(do_B, name.upper()),
            do_B,
            moment_matching_guidance,
            em_sampler,
        ),
        (
            "Do(B={}) - {}, MF-G, EM".format(do_B, name.upper()),
            do_B,
            manifold_guidance,
            em_sampler,
        ),
        (
            "Do(B={}) - {}, MM-G, EDM".format(do_B, name.upper()),
            do_B,
            moment_matching_guidance,
            edm_sampler,
        ),
        (
            "Do(B={}) - {}, MF-G, EDM".format(do_B, name.upper()),
            do_B,
            manifold_guidance,
            edm_sampler,
        ),
    ]:
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
        plot_samples(
            title,
            gen_A,
            gen_B,
            gen_C,
            figs_dir,
            sampling_time=(end_time - start_time),
            is_showing=False,
        )


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
