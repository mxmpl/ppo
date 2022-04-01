"""Agent interface.
"""
import functools

import acme
import chex
import dm_env
import jax
import numpy as np
import optax
from acme import specs
from acme.jax import utils
from acme.tf import savers
from acme.utils import counting, loggers
from jax import random

from ppo.advantages import compute_advantage_estimates
from ppo.base import Configuration
from ppo.learner import PPOLearner
from ppo.loss import ppo_loss
from ppo.networks import make_networks, make_policy_network
from ppo.replay import make_replay_buffer


class RandomAgent(acme.Actor):
    def __init__(self, environment_spec: specs.EnvironmentSpec) -> None:
        """Random agent.

        Parameters
        ----------
        environment_spec : specs.EnvironmentSpec
            Environment specifications.
        """
        self._action_spec = environment_spec.actions

    def select_action(self, observation: chex.Array) -> chex.Array:
        """Random action.

        Parameters
        ----------
        observation : chex.Array
            Observation of the environment.

        Returns
        -------
        chex.Array
            Sample from the standard normal distribution.
        """
        return np.random.randn(*self._action_spec.shape)


class PPOAgent(acme.Actor):
    def __init__(self, environment_spec: specs.EnvironmentSpec,
                 config: Configuration, workdir: str = './logs',
                 backend: str = 'cpu'):
        """PPO Agent.
        The loss function and the policy network used for inference are
        jitted.
        The optimizer is not the standard optax.adam because
        of the gradient normalization.

        Parameters
        ----------
        environment_spec : specs.EnvironmentSpec
            Environment specifications.
        config : Configuration
            Agent configuration.
        workdir : str, optional
            Logging folder, by default './logs'.
        """
        # Internalise agent components.
        self._seed = config.seed
        self._num_observations = 0
        self._key, learner_key = random.split(random.PRNGKey(self._seed))
        self._current_extras = {}
        batch_size = config.num_minibatches * config.minibatch_size

        # Reverb replay
        extra_spec = {'log_prob': np.ones(shape=(1,), dtype=np.float32)}
        reverb_replay = make_replay_buffer(
            environment_spec=environment_spec,
            extra_spec=extra_spec,
            sequence_length=config.horizon+1,
            batch_size=batch_size
        )
        data_iterator = reverb_replay.data_iterator
        self._server = reverb_replay.server
        self._adder = reverb_replay.adder
        self._client = reverb_replay.client
        self._can_sample = reverb_replay.can_sample

        # Internalise the networks.
        networks = make_networks(environment_spec, config.policy_layer_sizes,
                                 config.value_layer_sizes)
        policy_network = make_policy_network(
            environment_spec, config.policy_layer_sizes)
        self._inference = jax.jit(policy_network.apply, backend=backend)

        dummy_obs = utils.add_batch_dim(
            utils.zeros_like(environment_spec.observations))
        self._policy_params_keys = policy_network.init(
            self._key, dummy_obs, self._key).keys()

        # Loss, advantages and optimizer.
        loss_partial = functools.partial(
            ppo_loss, networks=networks,
            clipping_epsilon=config.clipping_epsilon,
            value_coeff=config.value_coeff, entropy_coeff=config.entropy_coeff
        )
        loss = jax.jit(loss_partial, backend=backend)
        advantages_partial = functools.partial(
            compute_advantage_estimates, gae_lambda=config.gae_lambda
        )
        advantages = jax.vmap(advantages_partial, in_axes=0)
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_gradient_norm),
            optax.scale_by_adam(eps=config.adam_epsilon),
            optax.scale(-config.learning_rate)
        )

        # Logging facilities
        logger = loggers.AutoCloseLogger(
            loggers.CSVLogger(workdir, label='ppo_agent'))
        counter = counting.Counter()

        # Create the learner.
        self._learner = PPOLearner(
            networks=networks,
            obs_spec=environment_spec.observations,
            data_iterator=data_iterator,
            optimizer=optimizer,
            loss=loss,
            advantages=advantages,
            num_epochs=config.num_epochs,
            num_minibatches=config.num_minibatches,
            discount=config.discount,
            first_key=learner_key,
            backend=backend,
            logger=logger,
            counter=counter
        )

        self._checkpointer = savers.Checkpointer(
            {'learner': self._learner, 'counter': counter},
            time_delta_minutes=10,
            subdirectory='checkpoints',
            directory=workdir
        )

    def select_action(self, observation: chex.Array) -> chex.Array:
        """Samples from the policy and returns an action.

        Parameters
        ----------
        observation : chex.Array
            Observation of timestep data from the environment.

        Returns
        -------
        chex.Array
            Action taken by the agent.
        """
        observation = utils.add_batch_dim(observation)
        self._key, action_key = random.split(self._key)
        params = self._learner.get_variables(self._policy_params_keys)
        action, extras = self._inference(params, observation, action_key)
        action = utils.to_numpy(utils.squeeze_batch_dim(action))
        self._current_extras = extras
        return action

    def observe_first(self, timestep: dm_env.TimeStep):
        """Make a first observation from the environment.

        Parameters
        ----------
        timestep : dm_env.TimeStep
            First timestep.
        """
        self._adder.add_first(timestep)

    def observe(self, action: chex.Array, next_timestep: dm_env.TimeStep):
        """Make an observation of timestep data from the environment.

        Parameters
        ----------
        action : chex.Array
            Action taken in the environment.
        next_timestep : dm_env.TimeStep
            Timestep produced by the environment given the `action`.
        """
        self._num_observations += 1
        self._adder.add(action, next_timestep, extras=self._current_extras)

    def update(self):
        """Perform an update of the actor parameters from past observations."""
        while self._can_sample():
            self._learner.step()
        self._checkpointer.save()

    def save(self):
        """Save the agent."""
        self._checkpointer.save(force=True)
