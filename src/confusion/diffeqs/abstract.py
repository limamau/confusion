from abc import abstractmethod

from jaxtyping import Array


class AbstractDiffEq:
    @abstractmethod
    def mu(self, t: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sigma(self, t: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def perturbation(self, x0: Array, t: Array, x1: Array) -> Array:
        raise NotImplementedError
