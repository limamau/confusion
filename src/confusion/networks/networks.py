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


# limamau: how to differentiate the following two classes?
# is there a point to create these two sub-classes at all?
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


class AbstractCausalNetwork(AbstractNetwork):
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
