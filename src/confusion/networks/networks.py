from abc import abstractmethod
from typing import Any, Optional

import equinox as eqx
from jaxtyping import Array, Key


class AbstractNetwork(eqx.Module):
    @abstractmethod
    def __call__(
        self,
        x: Any,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        raise NotImplementedError


class AbstractNaiveNetwork(AbstractNetwork):
    @abstractmethod
    def __call__(
        self,
        x: Array,
        t: Array,
        c: Optional[Array],
        *,
        key: Optional[Key] = None,
    ) -> Array:
        raise NotImplementedError
