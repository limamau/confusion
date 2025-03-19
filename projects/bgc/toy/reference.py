import os
from typing import Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jaxtyping import Array, Key

from confusion.utils import get_and_create_figs_dir


def get_joint(
    num_samples: int, key: Key, do_B: Optional[float] = None
) -> tuple[Array, Array, Array]:
    # split keys
    key_A, key_B, key_C = jr.split(key, 3)

    # A
    A = jr.normal(key_A, (num_samples, 1))

    # B
    if do_B is None:
        B = A + 0.5 * jr.normal(key_B, (num_samples, 1))
    else:
        B = do_B * jnp.ones((num_samples, 1))

    # C
    C = B + 0.5 * jr.normal(key_C, (num_samples, 1))

    return A, B, C


def get_file_name(figs_dir: str, title: str) -> str:
    return os.path.join(figs_dir, title.replace(" ", "").replace(",", "_") + ".png")


def plot_samples(
    title: str,
    sam_A: Array,
    sam_B: Array,
    sam_C: Array,
    figs_dir: str,
    bins: int = 20,
    alpha: float = 0.5,
    xlim: Tuple[int, int] = (-6, 6),
    ylim: Tuple[int, int] = (0, 200),
    figsize: Tuple[float, float] = (4, 2.5),
    dpi: int = 200,
    is_showing: bool = True,
    is_saving: bool = True,
    show_stats: bool = True,
    sampling_time: Optional[float] = None,
) -> None:
    # stats
    mean_A = jnp.mean(sam_A)
    std_A = jnp.std(sam_A)
    mean_B = jnp.mean(sam_B)
    std_B = jnp.std(sam_B)
    mean_C = jnp.mean(sam_C)
    std_C = jnp.std(sam_C)

    # plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.hist(sam_A.flatten(), bins=bins, alpha=alpha, label="A")
    plt.hist(sam_B.flatten(), bins=bins, alpha=alpha, label="B")
    plt.hist(sam_C.flatten(), bins=bins, alpha=alpha, label="C")
    plt.title(title)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend()

    # add stats to plot
    if show_stats:
        stats_text = f"A: μ={mean_A:.2f}, σ={std_A:.2f}\n"
        stats_text += f"B: μ={mean_B:.2f}, σ={std_B:.2f}\n"
        stats_text += f"C: μ={mean_C:.2f}, σ={std_C:.2f}"

        if sampling_time is not None:
            stats_text = stats_text + f"\nsamp. time: {sampling_time:.1f}s"

        plt.annotate(
            stats_text,
            xy=(0.03, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )

    # show/save plot
    if is_showing:
        plt.show()

    if is_saving:
        file_name = get_file_name(figs_dir, title)
        fig.savefig(file_name)

    plt.close()


def main():
    NUM_SAMPLES = 1000
    DO_B = 1.0
    SEED = 5678
    NAME = "reference"

    key = jr.PRNGKey(SEED)
    figs_dir = get_and_create_figs_dir(__file__, NAME)
    for title, do_B in [
        ("No intervention - reference", None),
        ("Do(B={}) - reference".format(DO_B), DO_B),
    ]:
        key, subkey = jr.split(key)
        ref_A, ref_B, ref_C = get_joint(NUM_SAMPLES, subkey, do_B=do_B)
        plot_samples(title, ref_A, ref_B, ref_C, figs_dir)


if __name__ == "__main__":
    main()
