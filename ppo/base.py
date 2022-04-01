"""Base interfaces for storing the trajectories, loss and learner metrics,
training state and the configuration of the PPOAgent.
"""
import dataclasses
import json
from typing import Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax


@chex.dataclass
class Trajectory:
    """Interface for the computed trajectories.

    Parameters
    ----------
    observations : chex.Array
        Observations of the environment from the trajectory.
    actions : chex.Array
        Actions from the trajectory.
    advantages : chex.Array
        Advantages from the trajectory.
    target_values : chex.Array
        Target values.
    log_probs : chex.Array
        Log probabilities of the `actions` in the trajectory.
    values : chex.Array
        Values.
    """
    observations: chex.Array
    actions: jnp.ndarray
    advantages: jnp.ndarray
    target_values: jnp.ndarray
    values: jnp.ndarray
    log_probs: jnp.ndarray


@chex.dataclass
class TrainingState:
    """Interface for the training state of the PPOLearner.

    Parameters
    ----------
    params : hk.Params
        Networks parameters.
    opt_state : optax.OptState
        Adam optimizer state.
    key : chex.PRNGKey
        Current RNG key.
    step : int
        Number of learning steps already performed.
    """
    params: hk.Params
    opt_state: optax.OptState
    key: chex.PRNGKey
    step: int


@chex.dataclass
class LossOutputs:
    """Interface to store metrics computed by the loss function.

    Parameters
    ----------
    total_loss : chex.Array
        Total loss
    clip_loss : chex.Array
        Main objective to optimize.
        .. math::
            \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta),
            1 - \epsilon, 1 + \epsilon)\hat{A}_t).
    value_loss : chex.Array
        Squared-error loss on the value function.
        .. math:: (V_\theta(s_t) - V_t^\text{targ})^2.
    entropy_loss : chex.Array
        Entropy penalty to ensure sufficient exploration.
        Not used in the original paper for Mujoco environments.
    """
    total_loss: chex.Array
    clip_loss: chex.Array
    value_loss: chex.Array
    entropy_loss: chex.Array


@chex.dataclass
class LearnerOutputs:
    """Interface to store metrics computed by the PPOLearner during a
    SGD step. Contains aggregated loss metrics.

    Parameters
    ----------
    total_loss : jnp.float64
        Mean total loss.
    clip_loss : jnp.float64
        Mean squared-error loss on the value function.
    entropy_loss : jnp.float64
        Mean entropy penalty.
    """
    total_loss: jnp.float64
    clip_loss: jnp.float64
    value_loss: jnp.float64
    entropy_loss: jnp.float64

    @classmethod
    def from_loss_outputs(cls, loss_outputs: LossOutputs):
        return cls(**jax.tree_map(jnp.mean, loss_outputs))


@chex.dataclass
class Configuration:
    """Configuration interface for the PPO Agent.
    Collect a batch, of size `num_minibatches` * `minibatch_size`,
    of trajectories of `horizon` samples.

    Parameters
    ----------
    seed: int
        Random seed.
    policy_layer_sizes: Tuple[int]
        Layer sizes of the policy MLP, by default (64, 64).
    value_layer_sizes: Tuple[int]
        Layer sizes of the value function MLP, by default (64, 64).
    horizon: int
        Trajectory horizon. Number of collected samples before an update,
         by default 2048.
    learning_rate: float
        Adam optimizer initial learning rate, by default 3e-4.
    num_epochs: int
        Number of epochs of optimization on the sampled data to perform
        each policy update, by default 10.
    num_minibatches: int
        Number of minibatches to divide the full batch into, by default 32.
    minibatch_size: int
        Size of each minibatch, by default 64.
    discount: float
        Discount, by default 0.99.
    gae_lambda: float
        Generalized Advantage Estimation parameter, by default 0.95.
    clipping_epsilon: float
        Clipping parameter, by default 0.2.
    entropy_coeff: float
        Coefficient of the entropy penalty in the full loss, by default 0.
    value_coeff: float
        Coefficient of the squared-error loss on the value function
        in the full loss, by default 1.
        Not useful if the parameters are not shared between the
        policy and the value function networks as the two parts
        of the loss can be optimized independently.
    adam_epsilon: float
        Epsilon coefficient in the Adam optimizer, by default 1e-5.
    max_gradient_norm: float
        Maximum gradient norm, by default 0.5.
    """
    seed: int
    policy_layer_sizes: Tuple[int] = (64, 64)
    value_layer_sizes: Tuple[int] = (64, 64)
    horizon: int = 2048
    learning_rate: float = 3e-4
    num_epochs: int = 10
    num_minibatches: int = 32
    minibatch_size: int = 64
    discount: float = 0.99
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.2
    entropy_coeff: float = 0.
    value_coeff: float = 1.
    adam_epsilon: float = 1e-5
    max_gradient_norm: float = 0.5

    def save(self, out: str) -> None:
        with open(out, 'w') as f:
            json.dump(dataclasses.asdict(self), f, indent=2)
