import os
import shutil
from typing import Tuple

import equinox as eqx
import orbax.checkpoint as ocp
from optax import OptState

from .diffusion import AbstractDiffusionModel
from .networks import AbstractNetwork


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
            item_names=("network", "opt_state"),
        )

    # here the AbstractDiffusionModel class is pass as argument
    # and is also returned (different than save method)
    def restore(
        self,
        model: AbstractDiffusionModel,
        abstract_opt_state: OptState,
        step: int | None = None,
    ) -> Tuple[AbstractDiffusionModel, OptState]:
        # restore latest if step is not given
        if step is None:
            step = self.mngr.latest_step()

        # partition
        saveable_network, static_network = eqx.partition(
            model.network, eqx.is_inexact_array
        )
        saveable_opt_state, static_opt_state = eqx.partition(
            abstract_opt_state, eqx.is_inexact_array
        )

        # restore
        restored = self.mngr.restore(
            step,
            args=ocp.args.Composite(
                **{
                    "network": ocp.args.StandardRestore(saveable_network),  # pyright: ignore
                    "opt_state": ocp.args.StandardRestore(saveable_opt_state),  # pyright: ignore
                }
            ),
        )

        # combine
        network = eqx.combine(restored["network"], static_network)
        opt_state = eqx.combine(restored["opt_state"], static_opt_state)
        model = eqx.tree_at(lambda m: m.network, model, network)

        return model, opt_state

    # here the AbstractNetwork class is pass as argument
    # and is also returned (different than restore method)
    def save(self, step: int, network: AbstractNetwork, opt_state: OptState) -> None:
        self.mngr.save(
            step,
            args=ocp.args.Composite(
                **{
                    "network": ocp.args.StandardSave(
                        eqx.filter(network, eqx.is_inexact_array)  # pyright: ignore
                    ),
                    "opt_state": ocp.args.StandardSave(
                        eqx.filter(opt_state, eqx.is_inexact_array)  # pyright: ignore
                    ),
                }
            ),
        )
