import functools as ft
import time
from typing import Callable, Iterator, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key
from optax import GradientTransformation, OptState

from .checkpointing import Checkpointer
from .diffusion import AbstractDiffusionModel
from .networks import AbstractNetwork


# auxiliary functions #
def dataloader(
    data: Array, conds: Array | None, batch_size: int, *, key: Key
) -> Iterator[Tuple[Array, Array | None]]:
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            if conds is not None:
                yield data[batch_perm], conds[batch_perm]
            else:
                yield data[batch_perm], None
            start = end
            end = start + batch_size


def single_loss_fn(
    network: AbstractNetwork,
    model: AbstractDiffusionModel,
    x0: Array,
    t: Array,
    c: Array | None,
    key: Key,
) -> Array:
    noise_key, dropout_key = jr.split(key)
    x, noise, std = model.perturbation(x0, t, key=noise_key)
    pred = network(x, t, c, key=dropout_key)
    return model.weights(t) * jnp.mean((pred + noise / std) ** 2)


def batch_loss_fn(
    network: AbstractNetwork,
    model: AbstractDiffusionModel,
    data: Array,
    conds: Array | None,
    t1: float,
    key: Key,
) -> Array:
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    # low-discrepancy sampling over t to reduce variance
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)
    loss_fn = ft.partial(single_loss_fn, network, model)
    loss_fn = jax.vmap(loss_fn)
    return jnp.mean(loss_fn(data, t, conds, losskey))


@eqx.filter_jit
def make_step(
    network: AbstractNetwork,
    model: AbstractDiffusionModel,
    data: Array,
    conds: Array | None,
    t1: float,
    key: Key,
    opt_state: OptState,
    opt_update: Callable,
) -> Tuple[Array, AbstractNetwork, Key, OptState]:
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(network, model, data, conds, t1, key)
    updates, opt_state = opt_update(grads, opt_state)
    network = eqx.apply_updates(network, updates)
    key = jr.split(key, 1)[0]
    return loss, network, key, opt_state


# training function #
def train(
    model: AbstractDiffusionModel,
    opt: GradientTransformation,
    data: Array,
    num_steps: int,
    batch_size: int,
    t1: float,
    print_every: int,
    ckpter: Checkpointer,
    key: Key,
    conds: Array | None = None,
):
    # optax will update the floating-point JAX arrays in the network
    network = model.network
    opt_state = opt.init(eqx.filter(network, eqx.is_inexact_array))

    # prep for training
    train_key, loader_key = jr.split(key)
    total_value = 0
    total_size = 0

    # let's go!
    start_time = time.time()
    for step, (data, conds) in zip(
        range(num_steps), dataloader(data, conds, batch_size, key=loader_key)
    ):
        value, network, train_key, opt_state = make_step(
            network, model, data, conds, t1, train_key, opt_state, opt.update
        )
        total_value += value.item()
        total_size += 1

        # logging
        if (step % print_every) == 0 or step == num_steps - 1:
            elapsed_time = time.time() - start_time
            print(
                f"Step: {step}, "
                + f"Loss: {total_value / total_size}, "
                + f"Elapsed time: {elapsed_time:.2f}s",
                flush=True,
            )
            total_value = 0
            total_size = 0

        # checkpointing
        if step % ckpter.save_every == 0 or step == num_steps - 1:
            ckpter.save(step, network, opt_state)
    ckpter.mngr.wait_until_finished()
