import argparse
from time import time

import acme
import matplotlib.pyplot as plt
import numpy as np
from acme.utils import loggers
from ppo import (Configuration, InvertedDoublePendulumEnv, InvertedPendulumEnv,
                 PPOAgent, ReacherEnv, WalkerEnv)
from ppo.utils import episodes_rewards_per_step, evaluate, save_video


def main(args):
    num_steps = args.num_steps
    config = Configuration(
        seed=args.seed,
        horizon=args.horizon,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_minibatches=args.num_minibatches,
        minibatch_size=args.minibatch_size,
        discount=args.discount,
        gae_lambda=args.gae_lambda,
        clipping_epsilon=args.clipping_epsilon,
        entropy_coeff=args.entropy_coeff,
        value_coeff=args.value_coeff,
        adam_epsilon=args.adam_epsilon,
        max_gradient_norm=args.max_gradient_norm
    )
    NAME_TO_ENV = {
        'pendulum': InvertedPendulumEnv,
        'reacher': ReacherEnv,
        'double-pendulum': InvertedDoublePendulumEnv,
        'walker': WalkerEnv
    }
    environment = NAME_TO_ENV[args.env](args.seed)
    specs = acme.make_environment_spec(environment)

    workdir = args.workdir
    agent = PPOAgent(specs, config, workdir)
    logger = loggers.CSVLogger(workdir, label='loop')
    file_path = logger.file_path
    logger = loggers.AutoCloseLogger(logger)
    file_splits = file_path.split('/')
    video_path = f'{file_splits[0]}/{file_splits[1]}/video.mp4'
    fig_path = f'{file_splits[0]}/{file_splits[1]}/rewards.pdf'
    config_path = f'{file_splits[0]}/{file_splits[1]}/config.json'

    config.save(config_path)
    loop = acme.EnvironmentLoop(environment, agent, logger=logger)
    agent.save()

    t = time()
    loop.run(num_steps=num_steps)
    print('Time: ', time() - t)
    frames = evaluate(environment, agent, 10)
    save_video(np.array(frames), filename=video_path)

    x, y = episodes_rewards_per_step(file_path)
    plt.figure(figsize=(15, 10))
    plt.plot(x, y)
    plt.savefig(fig_path)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the PPO agent')
    parser.add_argument('seed', type=int, help='Random seed.')
    parser.add_argument('--horizon', type=int, default=2048,
                        help='Trajectory horizon.')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Adam optimizer initial learning rate.')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs of optim on the sampled data.')
    parser.add_argument('--num_minibatches', type=int, default=32,
                        help='Number of minibatches to divide the batch into.')
    parser.add_argument('--minibatch_size', type=int, default=64,
                        help='Size of each minibatch.')
    parser.add_argument('--discount', type=float, default=0.99,
                        help='Discount.')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE parameter')
    parser.add_argument('--clipping_epsilon', type=float, default=0.2,
                        help='Clipping parameter.')
    parser.add_argument('--entropy_coeff', type=float,
                        default=0., help='Entropy coeff in the loss.')
    parser.add_argument('--value_coeff', type=float, default=1.,
                        help='Value coeff in the loss.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-5,
                        help='Epsilon coeff in Adam.')
    parser.add_argument('--max_gradient_norm', type=float,
                        default=0.5, help='Max gradient norm.')
    parser.add_argument('--workdir', type=str,
                        default='runs', help='Working directory')
    parser.add_argument('--num_steps', type=int, default=1_000_000,
                        help='Number of steps.')
    parser.add_argument('--env', type=str, default='pendulum',
                        help='Environment name. \
                            Must be either "pendulum", "reacher", ')
    args = parser.parse_args()
    main(args)
