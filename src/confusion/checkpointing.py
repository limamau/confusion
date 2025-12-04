import os
import shutil
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import orbax.checkpoint as ocp
from optax import GradientTransformation, OptState

from confusion.models import AbstractModel


class Checkpointer:
    def __init__(
        self,
        saving_path: str,
        max_save_to_keep: int,
        save_every: int,
        erase: bool = False,
        saving_criteria: str = "recency",
    ):
        self.saving_path = saving_path
        self.save_every = save_every
        if saving_criteria == "recency":
            best_fn = None
        elif saving_criteria == "best":
            # just bypass as we're using evaluators
            # storing a float based on a best_fn
            # rather than a PyTree of metrics
            best_fn = lambda x: x
        else:
            raise ValueError(f"Unknown saving criteria: {saving_criteria}")
        self.saving_criteria = saving_criteria

        if erase:
            if os.path.exists(saving_path):
                shutil.rmtree(saving_path)
        os.makedirs(saving_path, exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_save_to_keep,
            save_interval_steps=save_every,
            best_fn=best_fn,
            # limamau: this could be more general
            best_mode="min",
        )
        self.mngr = ocp.CheckpointManager(
            saving_path,
            options=options,
            item_names=("model", "opt_state"),
        )

    def restore(
        self,
        abstract_model: AbstractModel,
        opt: GradientTransformation,
        step: int | None = None,
    ) -> Tuple[AbstractModel, OptState]:
        # restore latest/best if step is not given
        if step is None:
            if self.saving_criteria == "best":
                step = self.mngr.best_step()
            elif self.saving_criteria == "latest":
                step = self.mngr.latest_step()
            else:
                raise ValueError(f"Unknown saving criteria: {self.saving_criteria}")

        # partition
        saveable_model, static_model = eqx.partition(
            abstract_model, eqx.is_inexact_array
        )
        opt_state = opt.init(eqx.filter(abstract_model, eqx.is_inexact_array))

        # restore
        restored = self.mngr.restore(
            step,
            args=ocp.args.Composite(
                **{
                    "model": ocp.args.StandardRestore(saveable_model),  # pyright: ignore
                    "opt_state": ocp.args.StandardRestore(opt_state),  # pyright: ignore
                }
            ),
        )

        # combine
        model = eqx.combine(restored["model"], static_model)
        opt_state = restored["opt_state"]

        return model, opt_state

    def save(
        self,
        step: int,
        model: AbstractModel,
        opt_state: OptState,
        value: float,
    ) -> None:
        if jnp.isnan(value):
            value = jnp.inf
        self.mngr.save(
            step,
            args=ocp.args.Composite(
                **{
                    "model": ocp.args.StandardSave(
                        eqx.filter(model, eqx.is_inexact_array)  # pyright: ignore
                    ),
                    "opt_state": ocp.args.StandardSave(
                        eqx.filter(opt_state, eqx.is_inexact_array)  # pyright: ignore
                    ),
                }
            ),
            metrics=value,
        )
