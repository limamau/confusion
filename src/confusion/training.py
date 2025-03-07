import functools as ft
import time
from typing import Callable, Iterator, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import filter_jit
from jaxtyping import Array, Key
from optax import GradientTransformation, OptState

from .checkpointing import Checkpointer
from .diffusion import AbstractDiffusionModel


# auxiliary functions #
def dataloader(
    data: Array, conds: Optional[Array], batch_size: int, *, key: Key
) -> Iterator[Tuple[Array, Optional[Array]]]:
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
    model: AbstractDiffusionModel,
    x0: Array,
    t: Array,
    c: Optional[Array],
    key: Key,
) -> Array:
    noise_key, dropout_key = jr.split(key)
    mean, std = model.perturbation(x0, t, key=noise_key)
    # clip std to avoid division by zero
    std = jnp.maximum(std, 1e-5)
    noise = jr.normal(key, x0.shape)
    x = mean + std * noise
    pred = model.score(x, t, c, key=dropout_key)
    return model.weights_fn(t) * jnp.mean((pred + noise / std) ** 2)


def batch_loss_fn(
    model: AbstractDiffusionModel,
    data: Array,
    conds: Optional[Array],
    key: Key,
) -> Array:
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    # low-discrepancy sampling over t to reduce variance (limamau: is this really necessary?)
    t = jr.uniform(tkey, (batch_size,), minval=model.t0, maxval=model.t1 / batch_size)
    t = t + (model.t1 / batch_size) * jnp.arange(batch_size)
    loss_fn = ft.partial(single_loss_fn, model)
    loss_fn = jax.vmap(loss_fn)
    return jnp.mean(loss_fn(data, t, conds, losskey))  # pyright: ignore


@filter_jit
def make_step(
    model: AbstractDiffusionModel,
    data: Array,
    conds: Optional[Array],
    key: Key,
    opt_state: OptState,
    opt_update: Callable,
) -> Tuple[Array, AbstractDiffusionModel, Key, OptState]:
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, data, conds, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state


# training function #
def train(
    model: AbstractDiffusionModel,
    opt: GradientTransformation,
    data: Array,
    num_steps: int,
    batch_size: int,
    print_every: int,
    ckpter: Checkpointer,
    key: Key,
    conds: Optional[Array] = None,
):
    # optax will update the floating-point JAX arrays in the model
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    # prep for training
    train_key, loader_key = jr.split(key)
    total_value = 0
    total_size = 0

    # training loop
    start_time = time.time()
    for step, (data, conds) in zip(
        range(num_steps + 1), dataloader(data, conds, batch_size, key=loader_key)
    ):
        value, model, train_key, opt_state = make_step(
            model, data, conds, train_key, opt_state, opt.update
        )
        total_value += value.item()
        total_size += 1

        # logging
        if step % print_every == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Step: {step}, "
                + f"Loss: {total_value / total_size:.4e}, "
                + f"Elapsed time: {elapsed_time:.2f}s",
                flush=True,
            )
            total_value = 0
            total_size = 0

        # checkpointing
        if step % ckpter.save_every == 0:
            ckpter.save(step, model, opt_state)
    ckpter.mngr.wait_until_finished()
