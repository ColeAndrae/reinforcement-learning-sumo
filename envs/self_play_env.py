"""SelfPlaySumoEnv v4 — No mirroring needed (egocentric obs)."""

import numpy as np
import gymnasium as gym
from .sumo_env import SumoEnv, heuristic_aggressive, heuristic_flanker, heuristic_positional
import random as R


class SelfPlaySumoEnv(gym.Env):
    def __init__(self, **kw):
        super().__init__()
        self._opp_fn = heuristic_aggressive
        self._env = SumoEnv(opponent_policy=self._dispatch)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._pool = []
        self._max_pool = 10
        self._progress = 0.0

    def set_training_progress(self, p):
        self._progress = max(0, min(1, p))

    def _dispatch(self, obs):
        return int(self._opp_fn(obs))

    def reset(self, seed=None, options=None):
        self._pick()
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(int(action))

    def update_opponent(self, model):
        def fn(obs):
            a, _ = model.predict(obs, deterministic=True)
            return int(a)
        self._pool.append(fn)
        if len(self._pool) > self._max_pool:
            self._pool.pop(0)

    def _pick(self):
        p = self._progress
        heuristics = [heuristic_aggressive, heuristic_flanker, heuristic_positional]

        if p < 0.4 or not self._pool:
            self._opp_fn = R.choice(heuristics)
        elif p < 0.7:
            if R.random() < 0.5:
                self._opp_fn = R.choice(self._pool)
            else:
                self._opp_fn = R.choice(heuristics)
        else:
            if R.random() < 0.15:
                self._opp_fn = R.choice(heuristics)
            else:
                self._opp_fn = R.choice(self._pool)
