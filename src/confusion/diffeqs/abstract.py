from abc import abstractmethod

from jaxtyping import Array


class AbstractDiffEq:
    @abstractmethod
    def mu(self, t: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sigma(self, t: Array) -> Array:
        raise NotImplementedError
