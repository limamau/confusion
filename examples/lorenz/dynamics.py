from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from jaxtyping import Array, Key

SIGMA, BETA, RHO = jnp.float32(10.0), jnp.float32(8.0 / 3.0), jnp.float32(28.0)
RESCALE = jnp.float32(20.0)


def lorenz(x: Array):
    """Computes the derivative of the standard Lorenz system."""
    dx = SIGMA * (x[1] - x[0])
    dy = x[0] * (RHO - x[2]) - x[1]
    dz = x[0] * x[1] - BETA * x[2]
    return jnp.array([dx, dy, dz])


def rescaled_lorenz(t: Array, x: Array):
    """
    Computes the time derivative of the rescaled Lorenz system: F_tilde(x) = F(20x)/20.
    This preserves the chaotic dynamics while keeping state values in a smaller,
    more stable range for neural networks.
    """
    return lorenz(RESCALE * x) / RESCALE


@eqx.filter_jit
def kr4_step(func: Callable, t: Array, x: Array, h: float):
    """Performs a single step of the 4th-order Runge-Kutta method."""
    k1 = h * func(t, x)
    k2 = h * func(t + h / 2.0, x + k1 / 2.0)
    k3 = h * func(t + h / 2.0, x + k2 / 2.0)
    k4 = h * func(t + h, x + k3)
    return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def generate_trajectory(
    key: Key,
    dynamics: Callable,
    total_time: float = 10.0,
    burn_in_time: float = 3.0,
    num_samples: int = 63,
    integration_step: float = 0.01,
):
    """
    Generates a single, discretized trajectory from the Lorenz system.

    Args:
        key: A JAX random key.
        dynamics: An instance of the LorenzSystem.
        total_time: Total time to integrate (e.g., 10s).
        burn_in_time: Time to discard from the beginning (e.g., 3s).
        num_samples: The final number of timesteps in the output trajectory (e.g., 60).
        integration_step: The small step 'h' for the RK4 integrator.

    Returns:
        A JAX array of shape (num_samples, 3) representing the trajectory.
    """
    # sample initial condition from a standard normal distribution
    y0 = random.normal(key, shape=(3,))

    # integration loop
    def scan_step(carry, t):
        carry = kr4_step(dynamics, t, carry, integration_step)
        return carry, carry

    num_integration_steps = int(total_time / integration_step)
    ts = jnp.linspace(0.0, total_time, num_integration_steps + 1, dtype=jnp.float32)
    _, full_trajectory = jax.lax.scan(scan_step, y0, ts)

    # discard the burn-in period
    burn_in_steps = int(burn_in_time / integration_step)
    trajectory_after_burn_in = full_trajectory[burn_in_steps:]

    # downsample the trajectory to the desired number of timesteps
    indices = jnp.linspace(0, len(trajectory_after_burn_in) - 1, num_samples, dtype=int)

    return trajectory_after_burn_in[indices]


def generate_ts(
    total_time: float = 10.0,
    burn_in_time: float = 3.0,
    num_samples: int = 63,
    integration_step: float = 0.01,
    nconds: int = 3,
) -> Array:
    num_integration_steps = int(total_time / integration_step)
    ts = jnp.linspace(0.0, total_time, num_integration_steps + 1, dtype=jnp.float32)
    burn_in_steps = int(burn_in_time / integration_step)
    ts = ts[burn_in_steps:]
    indices = jnp.linspace(0, len(ts) - 1, num_samples, dtype=int)
    ts = ts[indices]
    return ts[nconds:]


def generate_dataset(
    key: Key,
    num_train: int = 4000,
    num_test: int = 100,
    val_split: float = 0.2,
):
    """Generates the full training, validation, and test datasets for the Lorenz benchmark."""
    print("generating Lorenz attractor dataset...")

    # create a vectorized version of the trajectory generator
    vmap_generate = jax.vmap(generate_trajectory, in_axes=(0, None))

    # generate data
    total_trajectories = num_train + num_test
    keys = random.split(key, total_trajectories)

    print(f"generating {total_trajectories} trajectories...")
    all_data = vmap_generate(keys, rescaled_lorenz)
    print("...generation complete!")

    # split and shuffle data
    test_data = all_data[num_train:]
    train_val_data = all_data[:num_train]
    val_size = int(num_train * val_split)
    shuffle_key, _ = random.split(key)
    perm = random.permutation(shuffle_key, num_train)
    train_val_data = train_val_data[perm]
    validation_data = train_val_data[:val_size]
    train_data = train_val_data[val_size:]

    # reshape
    train_data = train_data.transpose(0, 2, 1).astype(jnp.float32)
    validation_data = validation_data.transpose(0, 2, 1).astype(jnp.float32)
    test_data = test_data.transpose(0, 2, 1).astype(jnp.float32)

    return train_data, validation_data, test_data


def split_in_conds(data: Array, nconds: int = 3):
    """Get the first N steps of the trajectory as pre-condition."""
    data, conds = data[:, :, nconds:], data[:, :, :nconds]

    return data, conds


def main():
    # generate dataset (this float32/63 issue is weird...)
    main_key = random.PRNGKey(0)
    jax.config.update("jax_enable_x64", True)
    train_set, val_set, test_set = generate_dataset(main_key)
    jax.config.update("jax_enable_x64", False)
    print("type(train_set):", type(train_set))
    print("type(val_set):", type(val_set))
    print("type(test_set):", type(test_set))

    # shapes
    print("training set shape:", train_set.shape)
    print("validation set shape:", val_set.shape)
    print("test set shape:", test_set.shape)

    # plot
    _, axs = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        subplot_kw={"projection": "3d"},
    )
    data, conds = split_in_conds(train_set)
    axs[0].plot(data[0, 0], data[0, 1], data[0, 2])
    axs[0].scatter(
        data[0, 0, 0],
        data[0, 1, 0],
        data[0, 2, 0],
        color="C0",
        s=50,
    )
    axs[0].plot(conds[0, 0], conds[0, 1], conds[0, 2])
    axs[0].scatter(
        conds[0, 0, 0],
        conds[0, 1, 0],
        conds[0, 2, 0],
        color="C1",
        s=50,
    )
    axs[0].set_title("traj #0")
    axs[1].plot(data[1, 0], data[1, 1], data[1, 2], label="trajectory")
    axs[1].scatter(
        data[1, 0, 0],
        data[1, 1, 0],
        data[1, 2, 0],
        label="data start",
        color="C0",
        s=50,
    )
    axs[1].plot(conds[1, 0], conds[1, 1], conds[1, 2], label="conditions")
    axs[1].scatter(
        conds[1, 0, 0],
        conds[1, 1, 0],
        conds[1, 2, 0],
        label="cond start",
        color="C1",
        s=50,
    )
    axs[1].set_title("traj #1")
    axs[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
