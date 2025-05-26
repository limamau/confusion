import time
from typing import Callable, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.random as jr
from equinox import filter_jit
from jaxtyping import Array, Key
from optax import GradientTransformation, OptState

from .checkpointing import Checkpointer
from .diffusion import AbstractDiffusionModel
from .evaluation import AbstractEvaluator, BestEval, LossOnlyEvaluator
from .logging import AbstractLogger, PrintOnlyLogger
from .losses import AbstractLoss
from .schedules import AbstractTimeSchedule, LinearTimeSchedule
from .utils import dataloader


def update_ema(
    ema_model: AbstractDiffusionModel, model: AbstractDiffusionModel, ema_rate: float
) -> AbstractDiffusionModel:
    # get model parameters
    ema_params = eqx.filter(ema_model, eqx.is_array)
    params = eqx.filter(model, eqx.is_array)

    def ema_update(ema_p, p):
        return ema_rate * ema_p + (1 - ema_rate) * p

    # update model params
    new_ema_params = jax.tree_map(ema_update, ema_params, params)
    updated_ema_model = eqx.combine(new_ema_params, ema_model)

    return updated_ema_model


@filter_jit
def make_step(
    model: AbstractDiffusionModel,
    ema_model: AbstractDiffusionModel,
    data: Array,
    conds: Optional[Array],
    key: Key,
    opt_state: OptState,
    opt_update: Callable,
    loss: AbstractLoss,
    t0: float,
    t1: float,
    ema_rate: float,
    time_schedule: AbstractTimeSchedule,
) -> Tuple[Array, AbstractDiffusionModel, AbstractDiffusionModel, Key, OptState]:
    filtered_loss = eqx.filter_value_and_grad(loss)
    batch_size = data.shape[0]
    newkey, tkey, losskey = jr.split(key, 3)
    losskeys = jr.split(losskey, batch_size)
    times = time_schedule(t0, t1, batch_size, tkey)
    step_loss, grads = filtered_loss(model, data, times, conds, losskeys)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    ema_model = update_ema(ema_model, model, ema_rate)
    return step_loss, model, ema_model, newkey, opt_state


def train(
    model: AbstractDiffusionModel,
    opt: GradientTransformation,
    loss: AbstractLoss,
    train_data: Array,
    eval_data: Union[Array, Tuple[Array, ...]],
    # train int args
    num_train_steps: int,
    train_batch_size: int,
    print_loss_every: int,
    # eval int args
    eval_batch_size: int,
    eval_every: int,
    # end int args
    key: Key,
    train_conds: Optional[Array] = None,
    eval_conds: Optional[Array] = None,
    t0: float = 1e-5,
    t1: float = 1.0,
    ema_rate: float = 0.99,
    ckpter: Optional[Checkpointer] = None,  # limamau: add abstraction to checkpointer
    evaluator: AbstractEvaluator = LossOnlyEvaluator(),
    logger: AbstractLogger = PrintOnlyLogger(),
    time_schedule: AbstractTimeSchedule = LinearTimeSchedule(),
) -> Optional[BestEval]:
    # let's go
    logger.log_msg(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # prep for checkpointing
    if ckpter is not None:
        if ckpter.saving_criteria == "best":
            assert eval_every == ckpter.save_every, (
                "If saving based on best metric, `eval_every` must be equal to `save_every`"
            )

    # optax will update the floating-point JAX arrays in the model
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    # get copy of the model for ema
    ema_model = eqx.tree_at(lambda x: x, model, model)

    # prep for training
    key, train_key, train_loader_key = jr.split(key, 3)
    total_value, total_size = 0, 0
    train_dataloader = dataloader(
        train_data, train_conds, train_batch_size, key=train_loader_key
    )

    # prep for evaluating
    key, eval_key, eval_loader_key = jr.split(key, 3)
    eval_dataloader = evaluator.dataloader(
        eval_data, eval_conds, eval_batch_size, key=eval_loader_key
    )

    # training loop
    for step, (train_data_batch, train_conds_batch) in zip(
        range(num_train_steps + 1), train_dataloader
    ):
        value, model, ema_model, train_key, opt_state = make_step(
            model,
            ema_model,
            train_data_batch,
            train_conds_batch,
            train_key,
            opt_state,
            opt.update,
            loss,
            t0,
            t1,
            ema_rate,
            time_schedule,
        )
        total_value += value.item()
        total_size += 1

        # logging
        if step % print_loss_every == 0:
            logger.log_step(step, total_value / total_size, "Train loss")
            total_value = 0
            total_size = 0

        # evaluation
        # limamau: fix batch_size as the same for evaluation and training (?),
        # but add a num_samples_per_eval parameter to control the number of
        # samples used for evaluation (which must be higher than training)
        # this can be done with a for loop or a map with batch fixed, then
        # one can write a make_eval_step and differentiate it from make_train_step
        if step % eval_every == 0:
            # get key and batch to eval...
            eval_key, subkey = jr.split(eval_key)
            eval_data_batch, eval_conds_batch = next(eval_dataloader)

            # ... from loss
            evaluator.loss_eval(
                step,
                ema_model,
                eval_data_batch,
                eval_conds_batch,
                loss,
                t0,
                t1,
                time_schedule,
                logger,
                key=subkey,
            )

            # ... from sampling
            evaluator.sampling_eval(
                step,
                ema_model,
                eval_data_batch,
                eval_conds_batch,
                logger,
                key=subkey,
            )

        # checkpointing
        if ckpter is not None and step % ckpter.save_every == 0:
            ckpter.save(step, ema_model, opt_state, evaluator.current_eval.value)
    if ckpter is not None:
        ckpter.mngr.wait_until_finished()

    if evaluator is not None:
        return evaluator.best_eval
