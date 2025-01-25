import jax
import equinox as eqx
import functools as ft
import jax.numpy as jnp
import jax.random as jr


# training function #
def train(
    num_steps,
    data,
    batch_size,
    network,
    weight,
    int_beta,
    t1,
    opt,
    print_every,
    ckpter,
    key,
    conds=None,
):
    # optax will update the floating-point JAX arrays in the model
    opt_state = opt.init(eqx.filter(network, eqx.is_inexact_array))
    
    # prep for training
    train_key, loader_key = jr.split(key)
    total_value = 0
    total_size = 0
    
    # training loop
    for step, (data, conds) in zip(
        range(num_steps), 
        dataloader(data, conds, batch_size, key=loader_key)
    ):
        value, network, train_key, opt_state = make_step(
            network, weight, int_beta, data, conds, t1, train_key, opt_state, opt.update
        )
        total_value += value.item()
        total_size += 1
        
        # logging
        if (step % print_every) == 0 or step == num_steps - 1:
            print(f"Step={step} Loss={total_value / total_size}", flush=True)
            total_value = 0
            total_size = 0
        
        # checkpointing
        if step % ckpter.save_every == 0 or step == num_steps - 1:
            ckpter.save(step, network, opt_state)
    ckpter.mngr.wait_until_finished()


# auxiliary functions #
def dataloader(data, conds, batch_size, *, key):
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            if conds is not None:
                yield data[batch_perm], conds[batch_perm]
            else:
                yield data[batch_perm], None
            start = end
            end = start + batch_size


def single_loss_fn(model, weight, int_beta, data, t, c, key):
    mean = data * jnp.exp(-0.5 * int_beta(t))
    var = jnp.maximum(1 - jnp.exp(-int_beta(t)), 1e-5)
    std = jnp.sqrt(var)
    noise_key, dropout_key = jr.split(key)
    noise = jr.normal(noise_key, data.shape)
    y = mean + std * noise
    pred = model(y, t, c, key=dropout_key)
    return weight(t) * jnp.mean((pred + noise / std) ** 2)


def batch_loss_fn(model, weight, int_beta, data, conds, t1, key):
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    # low-discrepancy sampling over t to reduce variance
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)
    loss_fn = ft.partial(single_loss_fn, model, weight, int_beta)
    loss_fn = jax.vmap(loss_fn)
    return jnp.mean(loss_fn(data, t, conds, losskey))


@eqx.filter_jit
def make_step(model, weight, int_beta, data, conds, t1, key, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, weight, int_beta, data, conds, t1, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state
