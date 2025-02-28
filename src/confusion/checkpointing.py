import os
import shutil
from typing import Tuple

import equinox as eqx
import orbax.checkpoint as ocp
from optax import GradientTransformation, OptState

from .diffusion import AbstractDiffusionModel


class Checkpointer:
    def __init__(
        self,
        saving_path: str,
        max_save_to_keep: int,
        save_every: int,
        erase: bool = False,
    ):
        self.saving_path = saving_path
        self.save_every = save_every

        if erase:
            if os.path.exists(saving_path):
                shutil.rmtree(saving_path)
        os.makedirs(saving_path, exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_save_to_keep,
            save_interval_steps=save_every,
        )
        self.mngr = ocp.CheckpointManager(
            saving_path,
            options=options,
            item_names=("model", "opt_state"),
        )

    def restore(
        self,
        abstract_model: AbstractDiffusionModel,
        opt: GradientTransformation,
        step: int | None = None,
    ) -> Tuple[AbstractDiffusionModel, OptState]:
        # restore latest if step is not given
        if step is None:
            step = self.mngr.latest_step()

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
        self, step: int, model: AbstractDiffusionModel, opt_state: OptState
    ) -> None:
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
        )
