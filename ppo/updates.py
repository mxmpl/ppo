"""Provides the function to perform the full update based on a batch
of trajectories.
"""
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import optax

from ppo.base import LossOutputs, TrainingState, Trajectory


def full_update(
    grad_fn: Callable,
    optimizer: optax.GradientTransformation,
    state: TrainingState,
    batch: Trajectory,
    num_minibatches: int,
    num_epochs: int,
    batch_size: int
) -> Tuple[TrainingState, LossOutputs]:
    """Perform the full update.
    We can leverage jax.lax.scan to decompose the full update into
    smaller chunks: repeat epochs updates for `num_epochs` times,
    and in each of these update, update using each of the
    `num_minibatches` minibatches.
    The advtanges normlization is implement in the original
    code.

    Parameters
    ----------
    grad_fn : Callable
        Gradient function of the PPO Loss.
    optimizer : optax.GradientTransformation
        Optimizer.
    state : TrainingState
        Current training state.
    batch : Trajectory
        Batch of trajectories.
    num_minibatches : int
        Number of minibatches.
    num_epochs : int
        Number of epochs of optimization on the sampled trajectories.
    batch_size : int
        Batch size.

    Returns
    -------
    Tuple[TrainingState, LossOutputs]
        New training state and loss metrics.
    """
    def minibatch_update(carry: Tuple, minibatch: Trajectory):
        """Update using a single minibatch."""
        params, opt_state = carry
        advantages = ((minibatch.advantages -
                       jnp.mean(minibatch.advantages, axis=0)) /
                      (jnp.std(minibatch.advantages, axis=0) + 1e-8))
        # Computing the gradients using the minibatch
        gradients, metrics = grad_fn(params,
                                     minibatch.observations,
                                     minibatch.actions,
                                     jnp.squeeze(minibatch.log_probs),
                                     minibatch.target_values,
                                     advantages)
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), metrics

    def epoch_update(carry: Tuple, unused: Any):
        """One epoch of updates."""
        key, params, opt_state, batch = carry
        key, subkey = jax.random.split(key)
        # Shuffle the batch and split it into minibatches for
        # this epoch
        permutation = jax.random.permutation(subkey, batch_size)
        shuffled_batch = jax.tree_map(
            lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree_map(
            lambda x: jnp.reshape(
                x, [num_minibatches, -1] + list(x.shape[1:])),
            shuffled_batch)
        # Perform each update using the minibatches
        (params, opt_state), metrics = jax.lax.scan(
            minibatch_update, (params, opt_state), minibatches,
            length=num_minibatches)
        return (key, params, opt_state, batch), metrics

    carry = (state.key, state.params, state.opt_state, batch)
    (key, params, opt_state, _), metrics = jax.lax.scan(
        epoch_update, carry, (), length=num_epochs)

    new_state = TrainingState(params=params,
                              opt_state=opt_state,
                              key=key,
                              step=state.step+1)
    return new_state, metrics
