from confusion.sampling import edm_sampling_ts
from confusion.sdes import VarianceExploding, VariancePreserving


def main():
    # parameters definitions
    SIGMA_MIN = 0.1
    SIGMA_MAX = 0.5
    BETA_MIN_BAR = 0.1
    BETA_MAX_BAR = 0.12
    RHO = 7
    N = 10
    T0 = 0.1
    T1 = 5.0
    TOL = 1e-2

    ve = VarianceExploding(
        SIGMA_MIN,
        SIGMA_MAX,
    )
    ts = edm_sampling_ts(ve, rho=RHO, N=N, t0=T0, t1=T1)
    print("ts:", ts)
    assert len(ts) == N
    assert all(ts[i] >= ts[i + 1] for i in range(len(ts) - 1))
    assert ts[0] <= T1 + TOL and ts[0] >= T1 - TOL
    assert ts[-1] >= T0 - TOL and ts[-1] <= T0 + TOL

    vp = VariancePreserving(
        BETA_MIN_BAR,
        BETA_MAX_BAR,
    )
    ts = edm_sampling_ts(vp, rho=RHO, N=N, t0=T0, t1=T1)
    print("ts:", ts)
    assert len(ts) == N
    assert all(ts[i] >= ts[i + 1] for i in range(len(ts) - 1))
    assert ts[0] <= T1 + TOL and ts[0] >= T1 - TOL
    assert ts[-1] >= T0 - TOL and ts[-1] <= T0 + TOL


if __name__ == "__main__":
    main()
