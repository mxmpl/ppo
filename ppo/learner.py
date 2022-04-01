"""Learner interface used in the PPOAgent.
"""
import collections
from typing import Callable, Iterator, Mapping, Tuple

import acme
import chex
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import reverb
from acme.jax.utils import add_batch_dim, zeros_like
from acme.utils import counting, loggers

from ppo.base import LearnerOutputs, TrainingState, Trajectory
from ppo.updates import full_update


class PPOLearner(acme.Learner):

    _state: TrainingState

    def __init__(self,
                 networks: hk.Transformed,
                 obs_spec: Mapping[str, dm_env.specs.Array],
                 data_iterator: Iterator,
                 optimizer: optax.GradientTransformation,
                 loss: Callable,
                 advantages: Callable,
                 num_epochs: int,
                 num_minibatches: int,
                 discount: float,
                 first_key: chex.PRNGKey,
                 backend: str,
                 logger: loggers.Logger,
                 counter: counting.Counter
                 ) -> None:
        """PPO Learner.

        Parameters
        ----------
        networks : hk.Transformed
            Transformed policy and value function networks.
        obs_spec : Mapping[str, dm_env.specs.Array]
            Observations specifications.
        data_iterator : Iterator
            Data iterator of the replay server.
        optimizer : optax.GradientTransformation
            Adam optimizer with gradient normalization.
        loss : Callable
            Loss function, with partial arguments already given.
        advantages : Callable
            Function to compute the estimates of the advantages,
            with partial arguments already given.
        num_epochs : int
            Number of epochs of updates.
        num_minibatches : int
            Number of minibatches.
        discount : float
            Discount rate.
        backend: str
            JIT backend ('cpu' or 'gpu').
        first_key : chex.PRNGKey
            First RNG key.
        logger : loggers.Logger
            Logger.
        counter : counting.Counter
            Counter.
        """

        # Internalise agent components.
        self._iterator = data_iterator
        self._optimizer = optimizer
        self._loss = loss
        self._advantages = advantages

        # Internalise logging/counting objects.
        self._counter = counter
        self._logger = logger

        # Internalise the hyperparameters.
        self._first_key = first_key
        self._discount = discount
        self._num_minibatches = num_minibatches
        self._num_epochs = num_epochs

        key_network, key = jax.random.split(self._first_key)
        dummy_obs = add_batch_dim(zeros_like(obs_spec))
        params = networks.init(key_network, dummy_obs)
        opt_state = optimizer.init(params)

        self._state = TrainingState(
            params=params,
            opt_state=opt_state,
            key=key,
            step=0)

        def sgd_step(state: TrainingState, samples: reverb.ReplaySample
                     ) -> Tuple[TrainingState, LearnerOutputs]:
            """SGD step. It is implemented here to easily use
            some variables that would not be useful later.
            This function is then being jitted and internalized.
            Until the call to `full_update`, most of the code in this
            function is there to ensure that the shapes of the different
            tensors are correct.
            """
            data = samples.data
            observations, actions, rewards, termination, extra = (
                data.observation, data.action,
                data.reward, data.discount, data.extras)
            log_probs = extra['log_prob']
            discounts = termination * self._discount

            def get_values(params, observations) -> jnp.ndarray:
                o = jax.tree_map(lambda x: jnp.reshape(
                    x, [-1] + list(x.shape[2:])),
                    observations)
                _, values = networks.apply(params, o)
                values = jnp.reshape(values, rewards.shape[0:2])
                return values
            values = get_values(state.params, observations)

            advantages, target_values = self._advantages(
                rewards, discounts, values)
            observations, actions, log_probs, values = jax.tree_map(
                lambda x: x[:, :-1],
                (observations, actions, log_probs, values))
            trajectories = Trajectory(
                observations=observations,
                actions=actions,
                values=values,
                log_probs=log_probs,
                advantages=advantages,
                target_values=target_values
            )
            num_sequences = target_values.shape[0]
            num_steps = target_values.shape[1]
            batch_size = num_sequences * num_steps
            batch = jax.tree_map(lambda x: x.reshape(
                (batch_size,) + x.shape[2:]), trajectories)

            grad_fn = jax.grad(self._loss, has_aux=True)
            new_state, loss_outputs = full_update(
                grad_fn, self._optimizer, state, batch,
                self._num_minibatches, self._num_epochs, batch_size)
            outputs = LearnerOutputs.from_loss_outputs(loss_outputs)
            return new_state, outputs
        self.sgd_step = jax.jit(sgd_step, backend=backend)

    def step(self):
        """Does a step of SGD."""

        # Do a batch of SGD and update self._state accordingly.
        samples = next(self._iterator)
        self._state, outputs = self.sgd_step(self._state, samples)

        # Update our counts and record it.
        counts = self._counter.increment(steps=1)

        # Write logs.
        self._logger.write({**outputs, **counts})

    def get_variables(self, policy_keys: collections.abc.KeysView
                      ) -> hk.Params:
        """Policy network variables after a number of SGD steps.

        Parameters
        ----------
        policy_keys : collections.abc.KeysView
            Parameters names in the full `params`.

        Returns
        -------
        hk.Params
            Parameters of the policy network.
        """
        params = self._state.params
        return {k: params[k] for k in policy_keys}

    def save(self) -> TrainingState:
        return self._state

    def restore(self, state: TrainingState):
        self._state = state
