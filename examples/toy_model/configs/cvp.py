import os
from dataclasses import dataclass, field
from typing import Any, Optional

import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Bool, Key

from confusion.diffusion import VariancePreserving
from confusion.guidance import MomentMatchingGuidance
from confusion.networks import CausalMultiLayerPerceptron
from confusion.sampling import ODESampler


@dataclass
class Config:
    """Configuration for variance preserving diffusion model with causal network."""

    name: str = "cvp"

    # 1. keys
    seed: int = 5678
    data_key: Key = field(init=False)
    net_key: Key = field(init=False)
    train_key: Key = field(init=False)
    sample_key: Key = field(init=False)

    # 2. dataset
    num_samples: int = 10_000

    # 3. network
    num_variables: int = 3
    num_blocks: int = 3
    hidden_dim: int = 256
    temb_dim: int = 2
    projection_scale: float = 1.0
    num_heads: int = 1
    qkv_size: int = 8
    is_conditional: bool = False
    causal_mask: Bool[Array, "num_vars num_vars"] = field(init=False)
    network: CausalMultiLayerPerceptron = field(init=False)

    # 4. diffusion model
    t0: float = 0.1
    t1: float = 3.0
    model: VariancePreserving = field(init=False)

    # 5. optimization
    num_steps: int = 10_000
    lr: float = 1e-3
    batch_size: int = 16
    opt: optax.GradientTransformation = field(init=False)

    # 6. logging and checkpointing
    print_every: int = 1000
    max_save_to_keep: int = 1
    save_every: int = 5_000
    saving_path: str = field(init=False)

    # 7. sampling
    dt0: float = 0.01
    sample_size: int = 1000
    conds: Optional[Any] = None
    sampler: ODESampler = field(init=False)

    # 8. guidance
    do_B: float = 1.0
    const_matrix: Array = field(init=False)
    y: Array = field(init=False)
    guidance: MomentMatchingGuidance = field(init=False)

    def __post_init__(self):
        # 1. keys init
        key = jr.PRNGKey(self.seed)
        self.data_key, self.net_key, self.train_key, self.sample_key = jr.split(key, 4)

        # 2. dataset init
        # done during train/evaluate

        # 3. network init
        self.causal_mask = jnp.ones(
            (self.num_variables, self.num_variables), dtype=bool
        )
        self.network = CausalMultiLayerPerceptron(
            num_blocks=self.num_blocks,
            vars_dim=self.num_variables,
            hidden_dim=self.hidden_dim,
            temb_dim=self.temb_dim,
            projection_scale=self.projection_scale,
            t1=self.t1,
            causal_mask=self.causal_mask,
            num_heads=self.num_heads,
            qkv_size=self.qkv_size,
            key=self.net_key,
            is_conditional=self.is_conditional,
        )

        # 4. diffusion model init
        def int_beta_fn(t):
            return t

        # weight is taken to increase importance of noise near t=0
        def weight_fn(t):
            return 1 - jnp.exp(-int_beta_fn(t))

        self.model = VariancePreserving(
            self.network,
            weight_fn,
            self.t0,
            self.t1,
            int_beta_fn,
        )

        # 5. optimizer init
        self.opt = optax.adam(self.lr)

        # 6. saving path init
        self.saving_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"../checkpoints/{self.name}",
        )

        # 7. sampler init
        self.sampler = ODESampler(self.dt0, self.t1)

        # 8. guidance init
        self.const_matrix = jnp.array([[0.0, self.do_B, 0.0]])
        self.y = jnp.array([self.do_B])
        self.guidance = MomentMatchingGuidance(
            self.const_matrix,
            self.y,
        )


def get_config() -> Config:
    return Config()
