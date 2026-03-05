from abc import abstractmethod

from jaxtyping import Array

from confusion.diffeqs.abstract import AbstractDiffEq


class AbstractODE(AbstractDiffEq):
    @abstractmethod
    def mu(self, t: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sigma(self, t: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def vector_field(self, x: Array, t: Array, x0: Array) -> Array:
        raise NotImplementedError


class OTFlowMatching(AbstractODE):
    sigma_min: float

    def __init__(self, sigma_min: float = 1e-5):
        self.sigma_min = sigma_min

    def mu(self, t: Array) -> Array:
        return 1 - (1 - self.sigma_min) * t

    def sigma(self, t: Array) -> Array:
        return t

    def vector_field(self, x: Array, t: Array, x0: Array) -> Array:
        mu_t = self.mu(t)
        sigma_t = self.sigma(t)
        mu_prime = -(1 - self.sigma_min)
        sigma_prime = 1.0

        return mu_prime * x0 + (sigma_prime / (sigma_t + 1e-5)) * (x - mu_t * x0)
