import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jaxtyping import Array, Key


def get_joint(
    num_samples: int,
    sequence_length: int,
    key: Key,
    amplitude: float = 1.0,
    freq: float = 0.06,
    noise_std: float = 0.1,
) -> Array:
    keys = jr.split(key, num_samples)
    t = jnp.arange(sequence_length)

    def sample_fn(key):
        noise_key1, noise_key2 = jr.split(key)

        # sine wave
        x1 = amplitude * jnp.sin(freq * t) + noise_std * jr.normal(
            noise_key1, (sequence_length,)
        )

        # cosine wave
        x2 = amplitude * jnp.cos(freq * t) + noise_std * jr.normal(
            noise_key2, (sequence_length,)
        )

        return jnp.stack([x1, x2], axis=-1)

    data = jax.vmap(sample_fn)(keys)
    data = jnp.reshape(data, (num_samples, 2, sequence_length))

    return data


def main():
    key = jr.PRNGKey(0)
    num_samples = 100
    sequence_length = 10
    amplitude = 1.0
    freq = 0.06
    noise_std = 1.0

    data = get_joint(num_samples, sequence_length, key, amplitude, freq, noise_std)

    # timeseries plot
    plt.figure()
    timeseries_mean = jnp.mean(data, axis=0)
    upper = jnp.percentile(data, 95, axis=0)
    lower = jnp.percentile(data, 5, axis=0)
    plt.plot(timeseries_mean[0], label="sin")
    plt.plot(timeseries_mean[1], label="cos")
    plt.fill_between(range(sequence_length), lower[0], upper[0], alpha=0.2)
    plt.fill_between(range(sequence_length), lower[1], upper[1], alpha=0.2)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
