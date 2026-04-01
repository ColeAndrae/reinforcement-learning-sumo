"""
SelfPlaySumoEnv — wraps SumoEnv for self-play training.
v2: 9 actions, fixed numpy int handling.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .sumo_env import SumoEnv
import random


class SelfPlaySumoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self._opponent_fn = self._random_policy
        self._env = SumoEnv(
            opponent_policy=self._dispatch_opponent,
            render_mode=render_mode,
        )
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._opponent_pool = []
        self._max_pool_size = 10
        self._episodes_since_update = 0
        self._swap_every = 50
        self._use_random_prob = 0.15

    def _dispatch_opponent(self, obs):
        return self._opponent_fn(obs)

    def reset(self, seed=None, options=None):
        self._episodes_since_update += 1
        if (self._episodes_since_update >= self._swap_every
                and len(self._opponent_pool) > 0):
            self._episodes_since_update = 0
            self._pick_opponent()
        obs, info = self._env.reset(seed=seed, options=options)
        self._swapped = random.random() < 0.5
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
        def policy_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
        self._opponent_pool.append(policy_fn)
        if len(self._opponent_pool) > self._max_pool_size:
            self._opponent_pool.pop(0)
        self._pick_opponent()

    def set_opponent_from_model(self, model):
        def policy_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
        self._opponent_fn = policy_fn

    def _pick_opponent(self):
        if random.random() < self._use_random_prob or len(self._opponent_pool) == 0:
            self._opponent_fn = self._random_policy
        else:
            self._opponent_fn = random.choice(self._opponent_pool)

    @staticmethod
    def _random_policy(obs):
        return np.random.randint(0, 9)

    @staticmethod
    def _mirror_obs(obs):
        mirrored = obs.copy()
        mirrored[0] *= -1    # own_x
        mirrored[2] *= -1    # own_vx
        mirrored[4] *= -1    # opp_rel_x
        mirrored[6] *= -1    # opp_vx
        mirrored[10] = -mirrored[10]  # angle_to_opp
        mirrored[11] = -mirrored[11]  # angle_to_center
        return mirrored

    @staticmethod
    def _mirror_action(action):
        # Mirror x-axis: swap +x/-x, keep z, swap diagonals accordingly
        # 0=noop 1=+x 2=-x 3=+z 4=-z 5=+x+z 6=+x-z 7=-x+z 8=-x-z
        mirror_map = {0:0, 1:2, 2:1, 3:3, 4:4, 5:7, 6:8, 7:5, 8:6}
        return mirror_map.get(int(action), 0)

    @property
    def unwrapped_sumo(self):
        return self._env
