"""Provides utility functions to evaluate the agents.
"""
import base64
from typing import Optional, Tuple

import imageio
import numpy as np
import pandas as pd
from IPython.display import HTML

from ppo.agents import PPOAgent
from ppo.baselines import symmetric_ema
from ppo.environments import GymEnv


def save_video(frames: np.ndarray, filename: str,
               frame_repeat: int = 1) -> HTML:
    """Save the video from the `frames`. Returns a rendered
    HTML output of the video.

    Parameters
    ----------
    frames : np.ndarray
        Time frames of the environment.
    filename : str
        Output video nameg.
    frame_repeat : int, optional
        How many time to repeat each frame in the video, by default 1.

    Returns
    -------
    HTML
        Rendered video.
    """
    with imageio.get_writer(filename, fps=60) as video:
        for frame in frames:
            for _ in range(frame_repeat):
                video.append_data(frame)
    video = open(filename, 'rb').read()
    b64_video = base64.b64encode(video)
    video_tag = ('<video  width="320" height="240" controls alt="test" '
                 'src="data:video/mp4;base64,{0}">').format(b64_video.decode())
    return HTML(video_tag)


def evaluate(environment: GymEnv, agent: PPOAgent,
             evaluation_episodes: int) -> np.ndarray:
    """Evaluate the `agent` on the `environment` for
    `evaluation_episodes` episodes. Prints the reward for each episode.

    Parameters
    ----------
    environment : GymEnv
        Environment.
    agent : PPOAgent
        Agent to evaluate.
    evaluation_episodes : int
        Number of evaluation episodes.

    Returns
    -------
    np.ndarray
        Rendered frames from the evaluation episodes.
    """
    frames = []
    for episode in range(evaluation_episodes):
        timestep = environment.reset()
        episode_return = 0
        steps = 0
        while not timestep.last():
            frames.append(environment.render())
            action = agent.select_action(timestep.observation)
            timestep = environment.step(action)
            steps += 1
            episode_return += timestep.reward
        print(f'Episode {episode} ended with reward '
              f'{episode_return} in {steps} steps')
    return np.array(frames)


def episodes_rewards_per_step(log_file: str
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns episodes rewards per step after having performed
    exponential moving average.

    Parameters
    ----------
    log_file : str
        Log file containing the `steps` and `epiosde_return`

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Times steps and episode rewards after symetric EMA.
    """
    df = pd.read_csv(log_file)
    steps, returns = np.array(df['steps']), np.array(df['episode_return'])
    x, y, _ = symmetric_ema(steps, returns, steps[0], steps[-1])
    return x, y
