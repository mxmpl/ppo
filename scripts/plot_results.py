"""Simple script to plot the results."""
import argparse
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ppo.baselines import symmetric_ema

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=str, default='pendulum',
                        help='Working directory, where the logs file are.')
    args = parser.parse_args()

    paths = glob(f'{args.workdir}/**/logs/loop/logs.csv')
    df = [pd.read_csv(p) for p in paths]
    steps = [np.array(d['steps']) for d in df]
    episode_return = [np.array(d['episode_return']) for d in df]
    z = [symmetric_ema(s, r, s[0], s[-1])
         for s, r in zip(steps, episode_return)]
    x, y, _ = zip(*z)
    x = np.array(x)
    y = np.array(y)
    std = y.std(axis=0)
    mean_y = y.mean(axis=0)
    mean_x = x.mean(axis=0)

    plt.figure(figsize=(15, 10))
    plt.plot(mean_x, mean_y)
    plt.fill_between(mean_x, mean_y-std, mean_y+std, alpha=0.3)
    plt.xlabel('Steps')
    plt.ylabel('Episode return (with EMA for smoothing)')
    plt.savefig(f'{args.workdir}/out.pdf')
    plt.savefig(f'{args.workdir}/out.png')
    plt.close()
