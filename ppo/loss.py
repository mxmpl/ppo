"""Provides the loss function.
"""
from typing import Tuple

import chex
import haiku as hk
import jax.numpy as jnp
import rlax

from ppo.base import LossOutputs


def ppo_loss(
    params: hk.Params,
    observations: chex.Array,
    actions: chex.Array,
    old_log_probs: chex.Array,
    target_values: chex.Array,
    advantages: chex.Array,
    *,
    networks: hk.Transformed,
    clipping_epsilon: float,
    value_coeff: float,
    entropy_coeff: float
) -> Tuple[chex.Array, LossOutputs]:
    """Loss function for the PPO agent.

    Parameters
    ----------
    params : hk.Params
        `networks` parameters.
    observations : chex.Array
        Observations of the environment from the trajectory.
    actions : chex.Array
        Actions from the trajectory.
    old_log_probs : chex.Array
        Log probabilities of the `actions` in the trajectory.
    target_values : chex.Array
        Target values.
    advantages : chex.Array
        Advantages from the trajectory.
    networks : hk.Transformed
        Transformed policy and value function networks.
    clipping_epsilon : float
        Clipping parameter.
    value_coeff : float
        Coefficient of the squared-error loss on the value function
        in the full loss.
    entropy_coeff: float
        Coefficient of the entropy penalty in the full loss.
    value_coeff: float
        Coefficient of the squared-error loss on the value function
        in the full loss.

    Returns
    -------
    Tuple[chex.Array, LossOutputs]
        Total loss and detailed metrics.
    """

    distribution, values = networks.apply(params, observations)
    log_probs = distribution.log_prob(actions)
    entropy = distribution.entropy()
    prob_ratios = jnp.exp(log_probs - old_log_probs)

    clip_loss = rlax.clipped_surrogate_pg_loss(
        prob_ratios, advantages, clipping_epsilon)
    value_loss = jnp.mean((target_values - values)**2)
    entropy_loss = - jnp.mean(entropy)
    total_loss = clip_loss + value_loss * value_coeff + \
        entropy_loss * entropy_coeff
    outputs = LossOutputs(
        total_loss=total_loss,
        clip_loss=clip_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss
    )
    return total_loss, outputs
