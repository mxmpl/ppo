"""Networks used by PPOAgent for Gym environments.
"""
from typing import Sequence, Tuple

import haiku as hk
import jax.numpy as jnp
import numpy as np
from acme import specs
from acme.jax import utils
from acme.jax.networks import MultivariateNormalDiagHead


def _policy_network(policy_layer_sizes: Tuple[int], num_dimensions: int
                    ) -> hk.Module:
    """Returns the policy network: a MLP with a head predicting the
    mean and log standard deviation of a Gaussian distribution.
    Calling it returns a tfd.MultivariateNormalDiag.

    Parameters
    ----------
    policy_layer_sizes : Tuple[int]
        Layer sizes of the policy MLP.
    num_dimensions : int
        Number of dimensions in the actions space.

    Returns
    -------
    hk.Module
        Policy network.
    """
    return hk.Sequential([
        utils.batch_concat,
        hk.nets.MLP(policy_layer_sizes, activation=jnp.tanh),
        MultivariateNormalDiagHead(num_dimensions)
    ])


def _value_network(value_layer_sizes: Tuple[int]) -> hk.Module:
    """Returns the value function network: a standard MLP
    with tanh activation.

    Parameters
    ----------
    value_layer_sizes : Tuple[int]
        Layer sizes of the value function MLP.

    Returns
    -------
    hk.Module
        Value function network.
    """
    return hk.Sequential([
        utils.batch_concat,
        hk.nets.MLP(value_layer_sizes, activation=jnp.tanh),
        hk.Linear(1), lambda x: jnp.squeeze(x, axis=-1)
    ])


def make_policy_network(
        environment_spec: specs.EnvironmentSpec,
        policy_layer_sizes: Sequence[int] = (64, 64)
) -> hk.Transformed:
    """Make a transformed policy network. Applying it returns
    the sampled actions and their log probabilities.
    Used for inference in the PPOAgent.

    Parameters
    ----------
    environment_spec : specs.EnvironmentSpec
        Environment specifications.
    policy_layer_sizes : Sequence[int], optional
        Layer sizes of the policy MLP, by default (64, 64).

    Returns
    -------
    hk.Transformed
        Transformed policy network.
    """
    num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

    def forward_fn(inputs, key):
        distribution = _policy_network(
            policy_layer_sizes, num_dimensions)(inputs)
        actions = distribution.sample(seed=key)
        log_prob = distribution.log_prob(actions)
        return actions, {'log_prob': log_prob}

    return hk.without_apply_rng(hk.transform(forward_fn))


def make_networks(
    environment_spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (64, 64),
    value_layer_sizes: Sequence[int] = (64, 64),
) -> hk.Transformed:
    """Make a transformed function returning the predicting
    distribution of the actions and the predicted value.

    Parameters
    ----------
    environment_spec : specs.EnvironmentSpec
        Environment specifications.
    policy_layer_sizes : Sequence[int], optional
        Layer sizes of the policy MLP, by default (64, 64).
    value_layer_sizes : Sequence[int], optional
        Layer sizes of the value function MLP, by default (64, 64).

    Returns
    -------
    hk.Transformed
        Transformed networks.
    """
    num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

    def forward_fn(inputs):
        policy_network = _policy_network(policy_layer_sizes, num_dimensions)
        value_network = _value_network(value_layer_sizes)
        action_distribution = policy_network(inputs)
        value = value_network(inputs)
        return (action_distribution, value)

    return hk.without_apply_rng(hk.transform(forward_fn))
