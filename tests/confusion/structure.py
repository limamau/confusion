from typing import Optional

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from confusion.diffeqs.odes import OTFlowMatching
from confusion.diffeqs.sdes import VariancePreserving
from confusion.guidance import GuidanceFree
from confusion.models.diffusion import StandardDiffusionModel
from confusion.models.flow import StandardFlowMatching
from confusion.networks import AbstractNetwork
from confusion.sampling import (
    ConstantStepEulerMaruyamaSampler,
    ODEDiffraxSampler,
    ScheduledEulerMaruyamaSampler,
    ScheduledEulerSampler,
)
from confusion.weighting import DenoiserWeighting


class DummyNetwork(AbstractNetwork):
    def __call__(
        self,
        x: Array,
        t: Array,
        c: Optional[Array] = None,
        *,
        key: Optional[Key] = None,
    ) -> Array:
        return x * t


class DummyWeighting(DenoiserWeighting):
    def __call__(self, t: Array) -> Array:
        return t


def main():
    key = jr.PRNGKey(0)
    data_shape = (2,)

    # setup diffusion model
    network = DummyNetwork()
    sde = VariancePreserving(beta_min=0.1, beta_max=20.0)
    weighting = DummyWeighting(sde=sde, sigma_data=1.0)
    diff_model = StandardDiffusionModel(network, weighting, sde)

    # setup drift model
    ode = OTFlowMatching(sigma_min=1e-5)
    drift_model = StandardFlowMatching(network, ode)

    guidance = GuidanceFree()

    print("verifying diffusion model sampling...")

    # test SDE samplers with diffusion model
    print("testing ConstantStepEulerMaruyamaSampler...")
    sampler = ConstantStepEulerMaruyamaSampler()
    _ = sampler.single_sample(diff_model, data_shape, guidance, None, None, key)
    print("   success!")

    print("testing ScheduledEulerMaruyamaSampler...")
    sampler = ScheduledEulerMaruyamaSampler(times=jnp.linspace(0, 1, 10))
    _ = sampler.single_sample(diff_model, data_shape, guidance, None, None, key)
    print("   success!")

    # test ODE samplers with diffusion model
    print("testing ScheduledEulerSampler...")
    sampler = ScheduledEulerSampler(times=jnp.linspace(0, 1, 10))
    _ = sampler.single_sample(diff_model, data_shape, guidance, None, None, key)
    print("   success!")

    print("testing ODEDiffraxSampler...")
    sampler = ODEDiffraxSampler(dt0=0.1)
    _ = sampler.single_sample(diff_model, data_shape, guidance, None, None, key)
    print("   success!")

    print("\nverifying drift model sampling...")

    # test ODE samplers with drift model
    print("testing ScheduledEulerSampler...")
    sampler = ScheduledEulerSampler(times=jnp.linspace(0, 1, 10))
    _ = sampler.single_sample(drift_model, data_shape, guidance, None, None, key)
    print("   success!")

    print("testing ODEDiffraxSampler...")
    sampler = ODEDiffraxSampler(dt0=0.1)
    _ = sampler.single_sample(drift_model, data_shape, guidance, None, None, key)
    print("   success!")

    # test SDE Samplers with drift model (should fail)
    print("testing ConstantStepEulerMaruyamaSampler...")
    try:
        sampler = ConstantStepEulerMaruyamaSampler()
        sampler.single_sample(drift_model, data_shape, guidance, None, None, key)
        print("   failed: should have raised NotImplementedError")
    except NotImplementedError:
        print("   success: raised NotImplementedError as expected")
    except Exception as e:
        print(f"   failed: raised unexpected error: {e}")


if __name__ == "__main__":
    main()
