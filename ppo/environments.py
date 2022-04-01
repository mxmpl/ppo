"""Wrappers for some Gym environments.
"""
import abc

import chex
import dm_env
import gym
import numpy as np
from acme import specs
from jax import random


class GymEnv(abc.ABC, dm_env.Environment):
    def __init__(self, seed: int = 0) -> None:
        """Generic wrapper for Gym environments."""
        self._env = gym.make(self.name)
        self._key = random.PRNGKey(seed)
        self._rng = np.random.default_rng(np.asarray(self._key))
        self._seeds = []

    @abc.abstractproperty
    def name(self) -> str:
        """Name of the Gym environment."""

    def step(self, action: chex.ArrayNumpy) -> dm_env.TimeStep:
        """Runs one timestep of the environment's dynamics."""
        new_obs, reward, done, _ = self._env.step(action)
        if done:
            return dm_env.termination(reward, new_obs)
        return dm_env.transition(reward, new_obs)

    def reset(self) -> dm_env.TimeStep:
        """Resets the environment to an initial state
        and returns an initial observation.
        """
        seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
        self._seeds.append(seed)
        obs = self._env.reset(seed=seed)
        return dm_env.restart(obs)

    def render(self):
        """Render the current screen, used for visualization."""
        return self._env.render(mode='rgb_array')

    def close(self) -> None:
        """Used to perform any necessary cleanup in a subclass."""
        self._env.close()


class InvertedPendulumEnv(GymEnv):
    """Inverted Pendulum environment."""

    @property
    def name(self) -> str:
        return 'InvertedPendulum-v2'

    def observation_spec(self) -> specs.Array:
        """State specifications: positional values of different
        body parts of the pendulum system, followed by the velocities
        of those individual parts (their derivatives) with all the
        positions ordered before all the velocities.

        Returns
        -------
        specs.Array
            Array of shape (4,) where the elements correspond to
            the following:
            0: position of the cart along the linear surface
            1: vertical angle of the pole on the cart
            2: linear velocity of the cart
            3: angular velocity of the pole on the cart
        """
        return specs.Array(shape=(4,), dtype=np.double)

    def action_spec(self) -> specs.BoundedArray:
        """Action specifications: numerical force applied to the cart,
        with magnitude representing the amount of force an
        sign representing the direction.

        Returns
        -------
        specs.Array
            Array of shape (1,) corresponding to the
            force applied to the cart.
        """
        return specs.BoundedArray(
            shape=(1,), minimum=-3., maximum=3., dtype=np.float32)


class InvertedDoublePendulumEnv(GymEnv):
    """Inverted Pendulum environment."""

    @property
    def name(self) -> str:
        return 'InvertedDoublePendulum-v2'

    def observation_spec(self) -> specs.Array:
        """State specifications: positional values of different
        body parts of the pendulum system, followed by the velocities
        of those individual parts (their derivatives) with all the
        positions ordered before all the velocities.

        Returns
        -------
        specs.Array
            Array of shape (4,) where the elements correspond to
            the following:
            0 position of the cart along the linear surface
            1 sine of the angle between the cart and the first pole
            2 sine of the angle between the two poles
            3 cosine of the angle between the cart and the first pole
            4 cosine of the angle between the two poles
            5 velocity of the car
            6 angular velocity of the angle between the cart and the first pole
            7 angular velocity of the angle between the two poles
            8 constraint force - 1
            9 constraint force - 2
            10 constraint force - 3
        """
        return specs.Array(shape=(11,), dtype=np.double)

    def action_spec(self) -> specs.BoundedArray:
        """Action specifications: numerical force applied to the cart,
        with magnitude representing the amount of force an
        sign representing the direction.

        Returns
        -------
        specs.Array
            Array of shape (1,) corresponding to the
            force applied to the cart.
        """
        return specs.BoundedArray(
            shape=(1,), minimum=-1., maximum=1., dtype=np.float32)


class ReacherEnv(GymEnv):
    """Reacher environment.
    “Reacher” is a two-jointed robot arm. The goal is to move the
    robot’s end effector (called fingertip) close to a target that
    is spawned at a random position.
    """

    @property
    def name(self) -> str:
        return 'Reacher-v2'

    def observation_spec(self) -> specs.BoundedArray:
        """Observations specifications: consist of
        - The cosine of the angles of the two arms
        - The sine of the angles of the two arms
        - The coordinates of the target
        - The angular velocities of the arms
        - The vector between the target and the reacher’s fingertip
        (3 dimensional with the last element being 0)

        Returns
        -------
        specs.BoundedArray
            Array of shape (11,) where the elements correspond to the
            following:
            0 	cosine of the angle of the first arm
            1 	cosine of the angle of the second arm
            2 	sine of the angle of the first arm
            3 	sine of the angle of the second arm
            4 	x-coorddinate of the target
            5 	y-coorddinate of the target
            6 	angular velocity of the first arm
            7 	angular velocity of the second arm
            8 	x-value of position_fingertip - position_target
            9 	y-value of position_fingertip - position_target
            10 	z-value of position_fingertip - position_target
                (0 since reacher is 2d and z is same for both)
        """
        return specs.Array(shape=(11,), dtype=np.double)

    def action_spec(self) -> specs.BoundedArray:
        """Action specifications: an action `(a, b)` represents the
        torques applied at the hinge joints.

        Returns
        -------
        specs.BoundedArray
            Array of shape (2,) where the elements correspond to
            the following:
            0: Torque applied at the first hinge
            (connecting the link to the point of fixture).
            1: Torque applied at the second hinge (connecting the two links)
        """
        return specs.BoundedArray(
            shape=(2,), minimum=-1., maximum=1., dtype=np.float32)


class WalkerEnv(GymEnv):
    """ Walker2d-v3 environment.
    The Walker has four main body parts: the torse, the two thighs
    below the torse, the two legs and the two feet. The goal is to coordinate
    the different body parts to make it able to walk by itself.
    It is done by applying torques on the six body parts.
    """

    @property
    def name(self) -> str:
        return 'Walker2d-v3'

    def observation_spec(self) -> specs.BoundedArray:
        """
        Positional values of the different body parts of the walker followed
        by the velocities of the different individual parts

        Returns
        -------
        specs.Array
        Array of shape (17,) where the elements correspond to:
            0: z-coordinate of the top (height of hopper)
            1: angle of the top
            2: angle of the thigh joint
            3: angle of the leg joint
            4: angle of the foot joint
            5: angle of the left thigh joint
            6: angle of the left leg joint
            7: angle of the left foot joint
            8: velocity of the x-coordinate of the top
            9: velocity of the z-coordinate (height) of the top
            10: angular velocity of the angle of the top
            11: angular velocity of the thigh hinge
            12: angular velocity of the leg hinge
            13: angular velocity of the foot hinge
            14: angular velocity of the thigh hinge
            15: angular velocity of the leg hinge
            16: angular velocity of the foot hinge
        """
        return specs.Array(shape=(17,), dtype=np.double)

    def action_spec(self) -> specs.BoundedArray:
        """
        An action represents the torques applied at the hinge joints.

        Returns
        -------
        specs.BoundedArray
        Array of shape (6,) where the elements correspond to
            the following:
            0: Torque applied on the thigh rotor (thigh joint).
            1: Torque applied on the leg rotor (leg joint).
            2: Torque applied on the foot rotor (foot joint).
            3: Torque applied on the left thigh rotor (left_thigh_joint).
            4: Torque applied on the left leg rotor (leg_left_joint).
            5: Torque applied on the left foot rotor (foot_left_joint).
        """
        return specs.BoundedArray(
            shape=(6,), minimum=-1., maximum=1., dtype=np.float32)
