import equinox as eqx
import orbax.checkpoint as ocp


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
            saving_path = ocp.test_utils.erase_and_create_empty(saving_path)
        
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_save_to_keep,
            save_interval_steps=save_every,
        )
        self.mngr = ocp.CheckpointManager(
            saving_path, options=options, item_names=('network', 'opt_state'),
        )
    
    def restore(self, abstract_network, abstract_opt_state, step=None):
        # restore latest if step is not given
        if step is None:
            step = self.mngr.latest_step()
        
        # partition
        saveable_network, static_network = eqx.partition(
            abstract_network, eqx.is_inexact_array
        )
        saveable_opt_state, static_opt_state = eqx.partition(
            abstract_opt_state, eqx.is_inexact_array
        )
        
        # restore
        restored = self.mngr.restore(
            step,
            args=ocp.args.Composite(
                network=ocp.args.StandardRestore(saveable_network),
                opt_state=ocp.args.StandardRestore(saveable_opt_state),
            ),
        )
        
        # combine
        network = eqx.combine(restored.network, static_network)
        opt_state = eqx.combine(restored.opt_state, static_opt_state)
        
        return network, opt_state
    
    def save(self, step, network, opt_state):
        self.mngr.save(
            step,
            args=ocp.args.Composite(
                network=ocp.args.StandardSave(
                    eqx.filter(network, eqx.is_inexact_array)
                ),
                opt_state=ocp.args.StandardSave(
                    eqx.filter(opt_state, eqx.is_inexact_array)
                ),
            ),
        )
