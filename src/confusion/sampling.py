import jax
import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


@eqx.filter_jit
def single_sample_fn(model, int_beta, data_shape, dt0, t1, conds, key):
    def drift(t, y, args):
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * (y + model(y, t, conds))

    term = dfx.ODETerm(drift)
    solver = dfx.Tsit5()
    t0 = 0
    y1 = jr.normal(key, data_shape)
    # reverse time, solve from t1 to t0
    sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, y1)
    return sol.ys[0]