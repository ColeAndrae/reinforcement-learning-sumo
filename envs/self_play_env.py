"""
SelfPlaySumoEnv v3 — Curriculum-based self-play.

Training phases:
  Phase 1 (0-30%):   Train vs mix of heuristic opponents
  Phase 2 (30-60%):  Train vs mix of heuristics + self-play pool
  Phase 3 (60-100%): Mostly self-play with occasional heuristic
"""

import numpy as np
import gymnasium as gym
from .sumo_env import (
    SumoEnv, OBS_DIM, ALL_HEURISTICS,
    heuristic_aggressive, heuristic_flanker, heuristic_positional, random_policy
)
import random as pyrandom


class SelfPlaySumoEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None):
        super().__init__()
        self._opponent_fn = heuristic_aggressive
        self._env = SumoEnv(
            opponent_policy=self._dispatch_opponent,
            render_mode=render_mode,
        )
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self._opponent_pool = []
        self._max_pool_size = 12
        self._total_episodes = 0

        # Curriculum settings
        self._training_progress = 0.0  # 0 to 1

    def set_training_progress(self, progress):
        """Call this from the training callback to update curriculum phase."""
        self._training_progress = max(0.0, min(1.0, progress))

    def _dispatch_opponent(self, obs):
        return int(self._opponent_fn(obs))

    def reset(self, seed=None, options=None):
        self._total_episodes += 1
        self._pick_opponent()

        obs, info = self._env.reset(seed=seed, options=options)

        # Randomly swap sides 50% of the time
        self._swapped = pyrandom.random() < 0.5
        if self._swapped:
            obs = self._mirror_obs(obs)

        return obs, info

    def step(self, action):
        action = int(action)
        if self._swapped:
            action = self._mirror_action(action)

        obs, reward, terminated, truncated, info = self._env.step(action)

        if self._swapped:
            obs = self._mirror_obs(obs)
            reward = -reward
            if info.get("winner") == 0:
                info["winner"] = 1
            elif info.get("winner") == 1:
                info["winner"] = 0

        return obs, reward, terminated, truncated, info

    def update_opponent(self, model):
        """Add current policy to opponent pool."""
        def policy_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
        self._opponent_pool.append(policy_fn)
        if len(self._opponent_pool) > self._max_pool_size:
            self._opponent_pool.pop(0)

    def set_opponent_from_model(self, model):
        def policy_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
        self._opponent_fn = policy_fn

    def _pick_opponent(self):
        """Curriculum-aware opponent selection."""
        p = self._training_progress
        pool_size = len(self._opponent_pool)

        if p < 0.3:
            # Phase 1: mostly heuristics, some random
            choices = ALL_HEURISTICS
            weights = [0.4, 0.25, 0.25, 0.1]  # aggressive, flanker, positional, random
        elif p < 0.6:
            # Phase 2: mix of heuristics and self-play
            if pool_size > 0 and pyrandom.random() < 0.5:
                self._opponent_fn = pyrandom.choice(self._opponent_pool)
                return
            else:
                choices = [heuristic_aggressive, heuristic_flanker, heuristic_positional]
                weights = [0.5, 0.25, 0.25]
        else:
            # Phase 3: mostly self-play
            if pool_size > 0 and pyrandom.random() < 0.8:
                # Bias toward recent opponents
                idx = pyrandom.choices(
                    range(pool_size),
                    weights=[i + 1 for i in range(pool_size)],  # linear bias toward newer
                    k=1
                )[0]
                self._opponent_fn = self._opponent_pool[idx]
                return
            else:
                choices = [heuristic_aggressive, heuristic_flanker]
                weights = [0.6, 0.4]

        self._opponent_fn = pyrandom.choices(choices, weights=weights, k=1)[0]

    @staticmethod
    def _mirror_obs(obs):
        """Mirror x-axis for swapped sides."""
        m = obs.copy()
        m[0] *= -1   # own_x
        m[2] *= -1   # own_vx
        m[4] *= -1   # opp_x
        m[6] *= -1   # opp_vx
        m[13] *= -1  # cross_product_sign
        return m

    @staticmethod
    def _mirror_action(action):
        # Swap +x/-x, keep z, swap diagonals
        mirror = {0:0, 1:2, 2:1, 3:3, 4:4, 5:7, 6:8, 7:5, 8:6}
        return mirror.get(action, 0)

    @property
    def unwrapped_sumo(self):
        return self._env
