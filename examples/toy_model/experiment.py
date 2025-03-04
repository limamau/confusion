import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt


def print_mean_and_variance(samples_A, samples_B, samples_C):
    mean_A = jnp.mean(samples_A)
    std_A = jnp.std(samples_A)
    mean_B = jnp.mean(samples_B)
    std_B = jnp.std(samples_B)
    mean_C = jnp.mean(samples_C)
    std_C = jnp.std(samples_C)
    print(f"A: mean={mean_A:.2f}, std={std_A:.2f}")
    print(f"B: mean={mean_B:.2f}, std={std_B:.2f}")
    print(f"C: mean={mean_C:.2f}, std={std_C:.2f}")
    print()


def get_joint(num_samples, key, do_B=None):
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


def main():
    # define experiment parameters
    NUM_SAMPLES = 1000
    DO_B = 3
    SEED = 5678

    # no intervention
    key = jr.PRNGKey(SEED)
    samples_A, samples_B, samples_C = get_joint(NUM_SAMPLES, key)
    print_mean_and_variance(samples_A, samples_B, samples_C)
    plt.hist(samples_A.flatten(), bins=20, alpha=0.5, label="A")
    plt.hist(samples_B.flatten(), bins=20, alpha=0.5, label="B")
    plt.hist(samples_C.flatten(), bins=20, alpha=0.5, label="C")
    plt.title("No Intervention")
    plt.legend()
    plt.show()

    # do(B) intervention
    samples_A, samples_B, samples_C = get_joint(NUM_SAMPLES, key, do_B=DO_B)
    print_mean_and_variance(samples_A, samples_B, samples_C)
    plt.hist(samples_A.flatten(), bins=20, alpha=0.5, label="A")
    plt.hist(samples_B.flatten(), bins=20, alpha=0.5, label="B")
    plt.hist(samples_C.flatten(), bins=20, alpha=0.5, label="C")
    plt.title("Do(B={})".format(DO_B))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
