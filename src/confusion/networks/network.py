from abc import abstractmethod
from typing import Optional

import equinox as eqx
from jaxtyping import Array, Key


class AbstractNetwork(eqx.Module):
    @abstractmethod
    def __call__(
        self,
        x: Array,
        t: Array,
        c: Array | None,
        *,
        key: Optional[Key] = None,
    ) -> Array:
        raise NotImplementedError
