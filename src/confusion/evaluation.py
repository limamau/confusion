from abc import abstractmethod
from typing import Iterator, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .diffusion import AbstractDiffusionModel
from .guidance import AbstractGuidance, GuidanceFree
from .logging import AbstractLogger
from .metrics import AbstractMetric
from .sampling import AbstractSampler
from .schedules import AbstractTimeSchedule
from .utils import batch_avg_loss
from .utils import dataloader as utilsdataloader


# helper eval classes #
class AbstractEval:
    step: int
    value: float

    def __init__(self, step: int = -1, value: float = jnp.inf):
        self.step = step
        self.value = value

    def __str__(self) -> str:
        return f"step: {self.step}, value: {self.value:.4e}"

    @abstractmethod
    def update(self, step: int, value: float) -> None:
        if value < self.value:
            self.step = step
            self.value = value


class BestEval(AbstractEval):
    best_mode: str

    def __init__(self, step: int = -1, value: float = jnp.inf, best_mode: str = "min"):
        super().__init__(step, value)
        assert best_mode in ["min", "max"], "best_mode must be 'min' or 'max'"
        self.best_mode = best_mode

    def update(self, step: int, value: float) -> None:
        if self.best_mode == "min":
            if value < self.value:
                self.step = step
                self.value = value
        elif self.best_mode == "max":
            if value > self.value:
                self.step = step
                self.value = value


class CurrentEval(AbstractEval):
    def __init__(self, step: int = -1, value: float = jnp.inf):
        super().__init__(step, value)

    def update(self, step: int, value: float) -> None:
        self.step = step
        self.value = value


# evaluator classes #
# limamau: some refactoring could be done here
class AbstractEvaluator:
    best_eval: BestEval
    current_eval: CurrentEval

    @staticmethod
    @abstractmethod
    def dataloader(
        data: Union[Array, Tuple[Array, ...]],
        conds: Optional[Array],
        batch_size: int,
        *,
        key: Key,
    ) -> Iterator[Tuple[Union[Array, Tuple[Array, ...]], Optional[Array]]]:
        raise NotImplementedError

    @abstractmethod
    def loss_eval(
        self,
        step: int,
        model: AbstractDiffusionModel,
        data: Union[Array, Tuple[Array, ...]],
        conds: Optional[Array],
        t0: float,
        t1: float,
        time_schedule: AbstractTimeSchedule,
        logger: AbstractLogger,
        *,
        key: Key,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def sampling_eval(
        self,
        step: int,
        model: AbstractDiffusionModel,
        data: Union[Array, Tuple[Array, ...]],
        conds: Optional[Array],
        logger: AbstractLogger,
        *,
        key: Key,
    ) -> None:
        raise NotImplementedError


class LossOnlyEvaluator(AbstractEvaluator):
    best_eval: BestEval
    current_eval: CurrentEval

    def __init__(self):
        self.best_eval = BestEval()
        self.current_eval = CurrentEval()

    @staticmethod
    def dataloader(
        data: Union[Array, Tuple[Array, ...]],
        conds: Optional[Array],
        batch_size: int,
        *,
        key: Key,
    ) -> Iterator[Tuple[Array, Optional[Array]]]:
        assert isinstance(data, Array)
        return utilsdataloader(data, conds, batch_size, key=key)

    def loss_eval(
        self,
        step: int,
        model: AbstractDiffusionModel,
        data: Union[Array, Tuple[Array, ...]],
        conds: Optional[Array],
        t0: float,
        t1: float,
        time_schedule: AbstractTimeSchedule,
        logger: AbstractLogger,
        *,
        key: Key,
    ) -> None:
        assert isinstance(data, Array)
        key, losskey = jr.split(key)
        batch_size = data.shape[0]
        losskeys = jr.split(losskey, batch_size)
        times = time_schedule(t0, t1, batch_size)
        value = batch_avg_loss(model, data, times, conds, losskeys).item()
        logger.log_step(step, value, "Eval loss", pre_str="+ ")

        # update evals
        self.current_eval.update(step, value)
        self.best_eval.update(step, value)

    # the following method essentially does nothing at all
    def sampling_eval(
        self,
        step: int,
        model: AbstractDiffusionModel,
        data: Union[Array, Tuple[Array, ...]],
        conds: Optional[Array],
        logger: AbstractLogger,
        *,
        key: Key,
    ) -> None:
        pass


class GuidanceFreeEvaluator(AbstractEvaluator):
    best_eval: BestEval
    current_eval: CurrentEval
    sample_size: int
    sampler: AbstractSampler
    metric: AbstractMetric

    def __init__(
        self,
        sampler: AbstractSampler,
        metric: AbstractMetric,
        best_mode: str = "min",
    ):
        self.best_eval = BestEval(best_mode=best_mode)
        self.current_eval = CurrentEval()
        self.sampler = sampler
        self.metric = metric

    @staticmethod
    def dataloader(
        data: Union[Array, Tuple[Array, ...]],
        conds: Optional[Array],
        batch_size: int,
        *,
        key: Key,
    ) -> Iterator[Tuple[Array, Optional[Array]]]:
        assert isinstance(data, Array)
        return utilsdataloader(data, conds, batch_size, key=key)

    def loss_eval(
        self,
        step: int,
        model: AbstractDiffusionModel,
        data: Union[Array, Tuple[Array, ...]],
        conds: Optional[Array],
        t0: float,
        t1: float,
        time_schedule: AbstractTimeSchedule,
        logger: AbstractLogger,
        *,
        key: Key,
    ) -> None:
        assert isinstance(data, Array)
        key, losskey = jr.split(key)
        batch_size = data.shape[0]
        losskeys = jr.split(losskey, batch_size)
        times = time_schedule(t0, t1, batch_size)
        value = batch_avg_loss(model, data, times, conds, losskeys).item()
        logger.log_step(step, value, "Eval loss", pre_str="+ ")

    def sampling_eval(
        self,
        step: int,
        model: AbstractDiffusionModel,
        data: Union[Array, Tuple[Array, ...]],
        conds: Optional[Array],
        logger: AbstractLogger,
        *,
        key: Key,
    ) -> None:
        if step == 0:
            return None

        assert isinstance(data, Array)

        gen_samples = self.sampler.sample(
            model=model,
            data_shape=data.shape[1:],
            pre_conds=conds,
            post_conds=None,
            key=key,
            num_samples=data.shape[0],
        )

        value = self.metric(gen_samples, data)
        logger.log_msg(f"  {self.metric.acronym}: {value:.4e}")

        # update evals
        self.current_eval.update(step, value.item())
        self.best_eval.update(step, value.item())


class MeanGuidancesAndMetricsEvaluator(AbstractEvaluator):
    best_eval: BestEval
    current_eval: CurrentEval
    sample_size: int
    sampler: AbstractSampler
    guidances: Tuple[AbstractGuidance, ...]
    metrics: Tuple[AbstractMetric, ...]

    def __init__(
        self,
        sampler: AbstractSampler,
        metrics: Tuple[AbstractMetric, ...],
        guidances: Tuple[AbstractGuidance, ...] = (GuidanceFree(),),
        best_mode: str = "min",
    ):
        self.best_eval = BestEval(best_mode=best_mode)
        self.current_eval = CurrentEval()
        self.sampler = sampler
        self.metrics = metrics
        self.guidances = guidances
        self.num_values = len(self.metrics) * len(self.guidances)

    @staticmethod
    def dataloader(
        data: Union[Array, Tuple[Array, ...]],
        conds: Optional[Array],
        batch_size: int,
        *,
        key: Key,
    ) -> Iterator[Tuple[Union[Array, Tuple[Array, ...]], Optional[Array]]]:
        dataset_size = data[0].shape[0]
        indices = jnp.arange(dataset_size)
        while True:
            key, subkey = jr.split(key, 2)
            perm = jr.permutation(subkey, indices)
            start = 0
            end = batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                if conds is not None:
                    yield (
                        tuple([data_i[batch_perm] for data_i in data]),
                        conds[batch_perm],
                    )
                else:
                    yield (
                        tuple([data_i[batch_perm] for data_i in data]),
                        None,
                    )
                start = end
                end = start + batch_size

    def loss_eval(
        self,
        step: int,
        model: AbstractDiffusionModel,
        data: Union[Array, Tuple[Array, ...]],
        conds: Optional[Array],
        t0: float,
        t1: float,
        time_schedule: AbstractTimeSchedule,
        logger: AbstractLogger,
        *,
        key: Key,
    ) -> None:
        assert isinstance(data, Tuple)
        assert len(data) == len(self.guidances)
        key, losskey = jr.split(key)
        batch_size = data[0].shape[0]
        losskeys = jr.split(losskey, batch_size)
        times = time_schedule(t0, t1, batch_size)
        value = batch_avg_loss(model, data[0], times, conds, losskeys).item()
        logger.log_step(step, value, "Eval loss", pre_str="+ ")

    def sampling_eval(
        self,
        step: int,
        model: AbstractDiffusionModel,
        data: Union[Array, Tuple[Array, ...]],
        conds: Optional[Array],
        logger: AbstractLogger,
        *,
        key: Key,
    ) -> None:
        if step == 0:
            return None

        values_sum = jnp.zeros(1)
        for i, guidance in enumerate(self.guidances):
            logger.log_msg(f"+ {guidance.__class__.__name__}:")
            gen_samples = self.sampler.sample(
                model=model,
                data_shape=data[i].shape[1:],
                pre_conds=conds,
                post_conds=None,
                key=key,
                num_samples=data[i].shape[0],
                guidance=guidance,
            )

            for metric in self.metrics:
                value = metric(gen_samples, data[i])
                logger.log_msg(f"  {metric.acronym}: {value:.4e}")
                values_sum += value

        mean_value = values_sum / self.num_values

        # update evals
        self.current_eval.update(step, mean_value.item())
        self.best_eval.update(step, mean_value.item())
