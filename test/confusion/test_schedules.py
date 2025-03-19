import jax.random as jr

from confusion.diffusion import VarianceExploding, VariancePreserving
from confusion.networks import MultiLayerPerceptron
from confusion.schedules import get_edm_sampling_ts


def main():
    # parameters definitions
    NUM_VARIABLES = 3
    HIDDEN_SIZE = 4
    SEED = 5678
    SIGMA_MIN = 0.1
    SIGMA_MAX = 0.5
    BETA_MIN_BAR = 0.1
    BETA_MAX_BAR = 0.12
    RHO = 7
    N = 10
    T0 = 0.1
    T1 = 5.0
    TOL = 1e-2

    # create models
    key = jr.PRNGKey(seed=SEED)
    network = MultiLayerPerceptron(NUM_VARIABLES, HIDDEN_SIZE, key=key)

    def dummy_weight_fn(dummy_t):
        return dummy_t

    ve = VarianceExploding(
        network,
        dummy_weight_fn,
        SIGMA_MIN,
        SIGMA_MAX,
    )

    vp = VariancePreserving(
        network,
        dummy_weight_fn,
        BETA_MIN_BAR,
        BETA_MAX_BAR,
    )

    # test ts function
    # for ve:
    ts = get_edm_sampling_ts(ve, rho=RHO, N=N, t0=T0, t1=T1)
    print("ts:", ts)
    assert len(ts) == N
    assert all(ts[i] >= ts[i + 1] for i in range(len(ts) - 1))
    assert ts[0] <= T1 + TOL and ts[0] >= T1 - TOL
    assert ts[-1] >= T0 - TOL and ts[-1] <= T0 + TOL
    # for vp:
    ts = get_edm_sampling_ts(vp, rho=RHO, N=N, t0=T0, t1=T1)
    print("ts:", ts)
    assert len(ts) == N
    assert all(ts[i] >= ts[i + 1] for i in range(len(ts) - 1))
    assert ts[0] <= T1 + TOL and ts[0] >= T1 - TOL
    assert ts[-1] >= T0 - TOL and ts[-1] <= T0 + TOL


if __name__ == "__main__":
    main()
