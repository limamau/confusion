from typing import Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Key

from .diffusion import AbstractDiffusionModel
from .guidance import AbstractGuidance, GuidanceFree
from .logging import Logger
from .metrics import AbstractMetric
from .sampling import AbstractSampler


class BestEval:
    step: int
    value: float

    def __init__(self, step: int = -1, value: float = jnp.inf):
        self.step = step
        self.value = value

    def __str__(self) -> str:
        return f"step: {self.step}, value: {self.value:.4e}"

    def update(self, step: int, value: float) -> None:
        if value < self.value:
            self.step = step
            self.value = value


# limamau: this could be more general
# maybe code an evaluate function and set print and plot as optional arguments,
# but this function will always update the best_eval_value. it could be also nice
# to set the metric for the best_eval_value average: median or mean or even pick one
class Evaluator:
    sample_size: int
    sampler: AbstractSampler
    guidances: Tuple[AbstractGuidance, ...]
    metrics: Tuple[AbstractMetric, ...]
    best_metric_idx: AbstractMetric
    best_eval: BestEval

    def __init__(
        self,
        sample_size: int,
        sampler: AbstractSampler,
        metrics: Tuple[AbstractMetric, ...],
        guidances: Tuple[AbstractGuidance, ...] = (GuidanceFree(),),
    ):
        self.sample_size = sample_size
        self.sampler = sampler
        self.metrics = metrics
        self.guidances = guidances
        self.num_values = len(self.metrics) * len(self.guidances)
        self.best_eval = BestEval()

    def evaluate(
        self,
        step: int,
        model: AbstractDiffusionModel,
        data: Array,
        conds: Optional[Array],
        norm_mean: Array,
        norm_std: Array,
        logger: Logger,
        *,
        key: Key,
    ) -> None:
        values_sum = jnp.zeros(1)

        for guidance in self.guidances:
            logger.log(f"+ {guidance.__class__.__name__}:")
            gen_samples = self.sampler.sample(
                model=model,
                data_shape=data.shape[1:],
                conds=conds,
                key=key,
                norm_mean=norm_mean,
                norm_std=norm_std,
                num_samples=self.sample_size,
                guidance=guidance,
            )

            for metric in self.metrics:
                value = metric(gen_samples, data)
                logger.log(f"  {metric.acronym}: {value:.4e}")
                values_sum += value

        mean_value = values_sum / self.num_values
        self.best_eval.update(step, mean_value.item())
