from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Key


class AbstractNetwork(eqx.Module):
    @abstractmethod
    def __call__(
        self, x: Array, t: Array, c: Array | None, *, key: Key | None = None
    ) -> Array:
        raise NotImplementedError
