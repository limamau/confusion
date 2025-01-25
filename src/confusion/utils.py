import jax.numpy as jnp


def denormalize(data, mean, std):
    return data * std + mean


def normalize(data, mean=None, std=None):
    if mean is None:
        mean = jnp.mean(data)
    
    if std is None:
        std = jnp.std(data)
    
    return (data - mean) / std, mean, std