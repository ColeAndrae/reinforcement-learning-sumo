#!/usr/bin/env python3
"""train.py v4 — curriculum training with validation."""

import os, sys, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from envs.self_play_env import SelfPlaySumoEnv
from envs.sumo_env import SumoEnv, heuristic_aggressive, heuristic_flanker, heuristic_positional

DEFAULTS = dict(timesteps=2_000_000, ckpt=100_000)


class CB(BaseCallback):
    def __init__(self, env, save_dir, total, ckpt_every):
        super().__init__(verbose=1)
        self.env, self.save_dir, self.total, self.ckpt_every = env, save_dir, total, ckpt_every
        self.next_ckpt = ckpt_every
        self.w = self.l = self.d = 0
        self.t0 = None

    def _on_training_start(self):
        self.t0 = time.time()
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self):
        self.env.set_training_progress(self.num_timesteps / self.total)
        for info in self.locals.get('infos', []):
            if 'winner' in info:
                if info['winner'] == 0: self.w += 1
                elif info['winner'] == 1: self.l += 1
                else: self.d += 1
        if self.num_timesteps >= self.next_ckpt:
            self.next_ckpt += self.ckpt_every
            p = os.path.join(self.save_dir, f'ckpt_{self.num_timesteps}')
            self.model.save(p)
            self.env.update_opponent(PPO.load(p))
            tot = self.w+self.l+self.d or 1
            phase = 'HEUR' if self.env._progress < 0.4 else ('MIX' if self.env._progress < 0.7 else 'SELF')
            print(f'  {self.num_timesteps:>10,} | {phase:>4} | WR:{self.w/tot:.0%} ({self.w}W/{self.l}L/{self.d}D) | Pool:{len(self.env._pool)} | {time.time()-self.t0:.0f}s')
            self.w = self.l = self.d = 0
        return True


def eval_vs(model, fn, name, n=100):
    env = SumoEnv(opponent_policy=fn)
    w = sum(1 for _ in range(n) if _play(model, env) == 0)
    print(f'  vs {name:>12}: {w}/{n} ({w}%)')
    return w/n

def _play(model, env):
    obs, _ = env.reset()
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, t, tr, info = env.step(int(a))
        done = t or tr
    return info.get('winner', -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=DEFAULTS['timesteps'])
    p.add_argument('--checkpoint-interval', type=int, default=DEFAULTS['ckpt'])
    a = p.parse_args()

    md = os.path.join(os.getcwd(), 'models')
    os.makedirs(md, exist_ok=True)

    print('=' * 60)
    print('  CUBE SUMO v4 — Egocentric Obs, Curriculum Training')
    print('=' * 60)
    print(f'  Steps: {a.timesteps:,}  |  Net: 128→128  |  Obs: 10 dims')
    print('=' * 60)

    env = SelfPlaySumoEnv()
    model = PPO('MlpPolicy', env, verbose=0,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.008,
        policy_kwargs=dict(net_arch=dict(pi=[128,128], vf=[128,128])))

    cb = CB(env, md, a.timesteps, a.checkpoint_interval)
    print('\nTraining...\n')
    model.learn(total_timesteps=a.timesteps, callback=cb)
    model.save(os.path.join(md, 'sumo_final'))

    print('\n' + '='*60 + '\n  EVALUATION\n' + '='*60)
    eval_vs(model, lambda o: np.random.randint(0,9), 'Random')
    eval_vs(model, heuristic_aggressive, 'Aggressive')
    eval_vs(model, heuristic_flanker, 'Flanker')
    eval_vs(model, heuristic_positional, 'Positional')
    print('\nDone! Run: python3 scripts/export_to_js.py && python3 scripts/validate.py')


if __name__ == '__main__':
    main()
