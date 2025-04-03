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
from .evaluation import BestEval, Evaluator
from .logging import Logger
from .losses import ScoreMatchingLoss
from .utils import normalize


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
    t0: float,
    t1: float,
) -> Tuple[Array, AbstractDiffusionModel, Key, OptState]:
    filtered_loss = eqx.filter_value_and_grad(loss)
    batch_size = data.shape[0]
    newkey, tkey, losskey = jr.split(key, 3)
    losskeys = jr.split(losskey, batch_size)

    # low-discrepancy sampling over time to reduce variance
    # at the end of the following two lines, times \in [t0, t1]
    # and the batches have times = {t, ..., t1}, t \in [t0, t1/batch_size]
    t = jr.uniform(tkey, (batch_size,), minval=t0, maxval=t1 / batch_size)
    times = t + (t1 / batch_size) * jnp.arange(batch_size)
    step_loss, grads = filtered_loss(model, data, times, conds, losskeys)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return step_loss, model, newkey, opt_state


def train(
    model: AbstractDiffusionModel,
    opt: GradientTransformation,
    loss: ScoreMatchingLoss,
    train_data: Array,
    eval_data: Array,
    # train int args
    num_train_steps: int,
    train_batch_size: int,
    print_loss_every: int,
    # eval int args
    eval_batch_size: int,
    eval_every: int,
    # end int args
    ckpter: Checkpointer,
    key: Key,
    train_conds: Optional[Array] = None,
    eval_conds: Optional[Array] = None,
    t0: float = 1e-5,
    t1: float = 1.0,
    evaluator: Optional[Evaluator] = None,
    logger: Optional[Logger] = None,
) -> Optional[BestEval]:
    # normalization
    train_data, train_mean, train_std = normalize(train_data)

    # optax will update the floating-point JAX arrays in the model
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    # prep for training
    key, train_key, train_loader_key = jr.split(key, 3)
    total_value = 0
    total_size = 0
    train_data_loader = dataloader(
        train_data, train_conds, train_batch_size, key=train_loader_key
    )

    # prep for evaluating
    key, eval_key, eval_loader_key = jr.split(key, 3)
    eval_data_loader = dataloader(
        eval_data, eval_conds, eval_batch_size, key=eval_loader_key
    )

    # prep for logging
    if logger is None:
        # default console-only logging
        logger = Logger()
    else:
        logger.log(f"Training log started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # training loop
    for step, (train_data_batch, train_conds_batch) in zip(
        range(num_train_steps + 1), train_data_loader
    ):
        value, model, train_key, opt_state = make_step(
            model,
            train_data_batch,
            train_conds_batch,
            train_key,
            opt_state,
            opt.update,
            loss,
            t0,
            t1,
        )
        total_value += value.item()
        total_size += 1

        # logging
        if step % print_loss_every == 0:
            logger.log_step(step, total_value / total_size, "Train loss")
            total_value = 0
            total_size = 0

        # evaluation
        # limamau: fix batch_size as the same for evaluation and training,
        # but add a num_samples_per_eval parameter to control the number of
        # samples used for evaluation (which must be higher than training)
        # this can be done with a for loop or a map with batch fixed, then
        # one can write a make_eval_step and differentiate it from make_train_step
        if step % eval_every == 0:
            eval_data_batch, eval_conds_batch = next(eval_data_loader)
            eval_key, losskey = jr.split(eval_key)
            losskeys = jr.split(losskey, eval_batch_size)
            # same principle as the low-discrepancy sampling over time,
            # but here we fix times[0] to t0 (of trainig, not sampling!)
            t0_batch = jnp.full(eval_batch_size, t0)
            times = t0_batch + (t1 / eval_batch_size) * jnp.arange(eval_batch_size)
            value = loss(model, eval_data_batch, times, eval_conds_batch, losskeys)

            # logging
            logger.log_step(step, value.item(), "Eval loss", pre_str="+ ")

            # additional evaluation using sampling and metrics
            if (evaluator is not None) and (step != 0):
                eval_key, subkey = jr.split(eval_key)
                evaluator.evaluate(
                    step,
                    model,
                    eval_data_batch,
                    eval_conds_batch,
                    train_mean,
                    train_std,
                    logger,
                    key=subkey,
                )

        # checkpointing
        if ckpter.save_every > 0 and step % ckpter.save_every == 0:
            ckpter.save(step, model, opt_state)
    ckpter.mngr.wait_until_finished()

    if evaluator is not None:
        return evaluator.best_eval
