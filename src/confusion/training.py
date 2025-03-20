import time
from typing import Callable, Iterator, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from equinox import filter_jit
from jaxtyping import Array, Key
from optax import GradientTransformation, OptState

from .checkpointing import Checkpointer
from .diffusion import AbstractDiffusionModel
from .losses import ScoreMatchingLoss


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


@filter_jit
def make_step(
    model: AbstractDiffusionModel,
    data: Array,
    conds: Optional[Array],
    key: Key,
    opt_state: OptState,
    opt_update: Callable,
    loss: ScoreMatchingLoss,
) -> Tuple[Array, AbstractDiffusionModel, Key, OptState]:
    filtered_loss = eqx.filter_value_and_grad(loss)
    step_loss, grads = filtered_loss(model, data, conds, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return step_loss, model, key, opt_state


def train(
    model: AbstractDiffusionModel,
    opt: GradientTransformation,
    loss: ScoreMatchingLoss,
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
            model, data, conds, train_key, opt_state, opt.update, loss
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
