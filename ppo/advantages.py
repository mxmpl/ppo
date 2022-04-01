"""Provides a function to compute the advantages with a
truncated version of generalized advantage estimation.
"""
from typing import Tuple

import jax
import jax.numpy as jnp
import rlax


def compute_advantage_estimates(
        rewards: jnp.ndarray, discounts: jnp.ndarray, values: jnp.ndarray,
        gae_lambda: float = 0.95) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the advantages with a truncated version of
    generalized advantage estimation, on fixed-length trajectory
    segments of length `horizon`.

    Parameters
    ----------
    rewards : jnp.ndarray
        Sequence of rewards.
    discounts : jnp.ndarray
        Sequence of discounts.
    values : jnp.ndarray
        Sequence of values.
    gae_lambda : float, optional
        Generalized Advantage Estimation parameter, by default 0.95.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Advantages and target values.
    """
    advantages = rlax.truncated_generalized_advantage_estimation(
        rewards[:-1], discounts[:-1], gae_lambda, values)
    advantages = jax.lax.stop_gradient(advantages)

    target_values = values[:-1] + advantages
    target_values = jax.lax.stop_gradient(target_values)
    return advantages, target_values
