"""
SelfPlaySumoEnv — wraps SumoEnv for self-play training.

The opponent is a frozen copy of the training agent, updated every
N episodes from a pool of past checkpoints. This prevents the agent
from overfitting to a single opponent strategy.

Usage:
    env = SelfPlaySumoEnv()
    model = PPO("MlpPolicy", env)
    # After K training steps, call env.update_opponent(model)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .sumo_env import SumoEnv
import random
import copy


class SelfPlaySumoEnv(gym.Env):
    """
    Wraps SumoEnv so the opponent is driven by a frozen policy
    from previous training checkpoints.
    """

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

        # Pool of past opponent policies (stores predict functions)
        self._opponent_pool = []
        self._max_pool_size = 10
        self._episodes_since_update = 0
        self._swap_every = 50  # swap opponent every N episodes
        self._use_random_prob = 0.2  # 20% chance of random opponent

    def _dispatch_opponent(self, obs):
        return self._opponent_fn(obs)

    def reset(self, seed=None, options=None):
        self._episodes_since_update += 1

        # Periodically pick a new opponent from the pool
        if (self._episodes_since_update >= self._swap_every
                and len(self._opponent_pool) > 0):
            self._episodes_since_update = 0
            self._pick_opponent()

        obs, info = self._env.reset(seed=seed, options=options)

        # Randomly swap sides 50% of the time for symmetry
        self._swapped = random.random() < 0.5
        if self._swapped:
            # Mirror observation
            obs = self._mirror_obs(obs)

        return obs, info

    def step(self, action):
        # Mirror action if sides are swapped
        if self._swapped:
            action = self._mirror_action(action)

        obs, reward, terminated, truncated, info = self._env.step(action)

        if self._swapped:
            obs = self._mirror_obs(obs)
            reward = -reward  # swap perspective
            if info.get("winner") == 0:
                info["winner"] = 1
            elif info.get("winner") == 1:
                info["winner"] = 0

        return obs, reward, terminated, truncated, info

    def update_opponent(self, model):
        """
        Add current model's policy to the opponent pool.
        Call this periodically during training.
        """
        # We store a callable that predicts actions deterministically
        # by cloning the current model's policy parameters.
        # Since SB3 models aren't trivially copyable, we store
        # a function that uses the model's predict() method.
        # The caller should save checkpoints and we load from those.

        # For simplicity, we just capture a reference to predict.
        # The training loop should call this with checkpoint-loaded models.
        def policy_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        self._opponent_pool.append(policy_fn)

        # Keep pool bounded
        if len(self._opponent_pool) > self._max_pool_size:
            self._opponent_pool.pop(0)

        self._pick_opponent()

    def set_opponent_from_model(self, model):
        """Set opponent directly from a SB3 model (for evaluation)."""
        def policy_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
        self._opponent_fn = policy_fn

    def _pick_opponent(self):
        """Select an opponent from the pool or use random."""
        if random.random() < self._use_random_prob or len(self._opponent_pool) == 0:
            self._opponent_fn = self._random_policy
        else:
            self._opponent_fn = random.choice(self._opponent_pool)

    @staticmethod
    def _random_policy(obs):
        return np.random.randint(0, 5)

    @staticmethod
    def _mirror_obs(obs):
        """Mirror observation for swapped sides."""
        mirrored = obs.copy()
        # Negate x-components (positions and velocities)
        mirrored[0] *= -1   # own_x
        mirrored[2] *= -1   # own_vx
        mirrored[4] *= -1   # opp_rel_x
        mirrored[6] *= -1   # opp_vx
        # Angles need adjustment
        mirrored[10] = -mirrored[10]  # angle_to_opp (flip sign for mirror)
        mirrored[11] = -mirrored[11]  # angle_to_center
        return mirrored

    @staticmethod
    def _mirror_action(action):
        """Mirror action for swapped sides."""
        mirror_map = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
        return mirror_map.get(int(action), 0)

    @property
    def unwrapped_sumo(self):
        return self._env
