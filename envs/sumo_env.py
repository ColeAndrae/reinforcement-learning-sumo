"""
SumoEnv — Gymnasium-compatible sumo wrestling environment.

v2: 9 discrete actions (cardinals + diagonals), stronger reward shaping.

Observation (12 dims per agent, relative frame):
  [0]  own_x / RING_RADIUS
  [1]  own_z / RING_RADIUS
  [2]  own_vx / MAX_SPEED
  [3]  own_vz / MAX_SPEED
  [4]  opp_rel_x / (2*RING_RADIUS)
  [5]  opp_rel_z / (2*RING_RADIUS)
  [6]  opp_vx / MAX_SPEED
  [7]  opp_vz / MAX_SPEED
  [8]  own_dist_to_edge / RING_RADIUS
  [9]  opp_dist_to_edge / RING_RADIUS
  [10] angle_to_opp / pi
  [11] angle_to_center / pi

Actions (9 discrete):
  0=noop  1=+x  2=-x  3=+z  4=-z  5=+x+z  6=+x-z  7=-x+z  8=-x-z
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

RING_RADIUS   = 5.5
RING_OUT_R    = RING_RADIUS - 0.2
CUBE_SIZE     = 0.9
CUBE_MASS     = 1.0
PUSH_FORCE    = 28.0
DIAG_FORCE    = PUSH_FORCE / math.sqrt(2)
FRICTION      = 0.92
BOUNCE        = 0.3
FIXED_DT      = 1.0 / 60.0
MAX_SPEED     = 15.0
MAX_STEPS     = 600
SUBSTEPS      = 2

ACTION_FORCES = [
    (0.0, 0.0),                          # 0: noop
    (PUSH_FORCE, 0.0),                   # 1: +x
    (-PUSH_FORCE, 0.0),                  # 2: -x
    (0.0, PUSH_FORCE),                   # 3: +z
    (0.0, -PUSH_FORCE),                  # 4: -z
    (DIAG_FORCE, DIAG_FORCE),            # 5: +x+z
    (DIAG_FORCE, -DIAG_FORCE),           # 6: +x-z
    (-DIAG_FORCE, DIAG_FORCE),           # 7: -x+z
    (-DIAG_FORCE, -DIAG_FORCE),          # 8: -x-z
]


class SumoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, opponent_policy=None, render_mode=None):
        super().__init__()
        self.opponent_policy = opponent_policy or self._random_policy
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(12,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.steps = 0
        self.agents = [np.zeros(4, dtype=np.float64) for _ in range(2)]
        self._prev_dist_between = 0.0
        self._prev_opp_edge_dist = 0.0
        self._prev_own_edge_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        jitter = self.np_random.uniform(-0.3, 0.3, size=2)
        self.agents[0] = np.array([-2.5 + jitter[0], 0.0 + jitter[1], 0.0, 0.0])
        self.agents[1] = np.array([ 2.5 - jitter[0], 0.0 - jitter[1], 0.0, 0.0])
        self.steps = 0

        dx = self.agents[1][0] - self.agents[0][0]
        dz = self.agents[1][1] - self.agents[0][1]
        self._prev_dist_between = math.sqrt(dx*dx + dz*dz)
        self._prev_opp_edge_dist = RING_OUT_R - math.sqrt(self.agents[1][0]**2 + self.agents[1][1]**2)
        self._prev_own_edge_dist = RING_OUT_R - math.sqrt(self.agents[0][0]**2 + self.agents[0][1]**2)

        return self._get_obs(0), {}

    def step(self, action):
        opp_obs = self._get_obs(1)
        opp_action = self.opponent_policy(opp_obs)
        actions = [int(action), int(opp_action)]

        terminated = False
        winner = -1

        for _ in range(SUBSTEPS):
            for i in range(2):
                fx, fz = ACTION_FORCES[actions[i]]
                self.agents[i][2] += fx / CUBE_MASS * FIXED_DT
                self.agents[i][3] += fz / CUBE_MASS * FIXED_DT

            for i in range(2):
                f = FRICTION ** (FIXED_DT * 60)
                self.agents[i][2] *= f
                self.agents[i][3] *= f

            self._resolve_collision()

            for i in range(2):
                self.agents[i][0] += self.agents[i][2] * FIXED_DT
                self.agents[i][1] += self.agents[i][3] * FIXED_DT

            for i in range(2):
                dist = math.sqrt(self.agents[i][0]**2 + self.agents[i][1]**2)
                if dist > RING_OUT_R:
                    terminated = True
                    winner = 1 - i
                    break
            if terminated:
                break

        self.steps += 1

        # ── Reward ──────────────────────────────────────────
        reward = 0.0

        if terminated:
            reward = 10.0 if winner == 0 else -10.0
        else:
            me = self.agents[0]
            opp = self.agents[1]

            own_dist = math.sqrt(me[0]**2 + me[1]**2)
            opp_dist = math.sqrt(opp[0]**2 + opp[1]**2)
            own_edge = RING_OUT_R - own_dist
            opp_edge = RING_OUT_R - opp_dist

            dx = opp[0] - me[0]
            dz = opp[1] - me[1]
            dist_between = math.sqrt(dx*dx + dz*dz) or 0.01

            # 1. Closing speed reward: positive when distance decreases
            closing_reward = (self._prev_dist_between - dist_between) * 0.15

            # 2. Opponent edge pressure: reward when opponent gets closer to edge
            opp_edge_delta = self._prev_opp_edge_dist - opp_edge
            edge_pressure = opp_edge_delta * 0.2

            # 3. Self-preservation: penalize moving toward own edge
            own_edge_delta = self._prev_own_edge_dist - own_edge
            self_preservation = own_edge_delta * -0.15

            # 4. Proximity bonus: reward being close to opponent (encourages engagement)
            proximity = max(0, 1.0 - dist_between / 4.0) * 0.01

            # 5. Positional advantage: reward when opponent is further from center than us
            positional = (opp_dist - own_dist) / RING_RADIUS * 0.02

            # 6. Danger zone penalty: strong penalty when very close to edge
            if own_edge < 1.0:
                danger_penalty = -0.05 * (1.0 - own_edge)
            else:
                danger_penalty = 0.0

            # 7. Time penalty
            time_penalty = -0.002

            reward = closing_reward + edge_pressure + self_preservation + proximity + positional + danger_penalty + time_penalty

            self._prev_dist_between = dist_between
            self._prev_opp_edge_dist = opp_edge
            self._prev_own_edge_dist = own_edge

        truncated = self.steps >= MAX_STEPS
        if truncated and not terminated:
            reward -= 2.0

        obs = self._get_obs(0)
        info = {"winner": winner, "steps": self.steps}
        return obs, reward, terminated, truncated, info

    def _resolve_collision(self):
        a, b = self.agents
        dx = b[0] - a[0]
        dz = b[1] - a[1]
        if abs(dx) < CUBE_SIZE and abs(dz) < CUBE_SIZE:
            dist = math.sqrt(dx*dx + dz*dz) or 0.01
            nx = dx / dist
            nz = dz / dist
            overlap = CUBE_SIZE - dist
            if overlap > 0:
                a[0] -= nx * overlap * 0.5
                a[1] -= nz * overlap * 0.5
                b[0] += nx * overlap * 0.5
                b[1] += nz * overlap * 0.5
            rel_v = (a[2] - b[2]) * nx + (a[3] - b[3]) * nz
            if rel_v > 0:
                impulse = rel_v * (1 + BOUNCE)
                a[2] -= impulse * nx * 0.5
                a[3] -= impulse * nz * 0.5
                b[2] += impulse * nx * 0.5
                b[3] += impulse * nz * 0.5

    def _get_obs(self, agent_idx):
        me = self.agents[agent_idx]
        opp = self.agents[1 - agent_idx]
        own_dist = math.sqrt(me[0]**2 + me[1]**2)
        opp_dist = math.sqrt(opp[0]**2 + opp[1]**2)
        rel_x = opp[0] - me[0]
        rel_z = opp[1] - me[1]
        angle_to_opp = math.atan2(rel_z, rel_x)
        angle_to_center = math.atan2(-me[1], -me[0])
        return np.array([
            me[0] / RING_RADIUS, me[1] / RING_RADIUS,
            me[2] / MAX_SPEED, me[3] / MAX_SPEED,
            rel_x / (2 * RING_RADIUS), rel_z / (2 * RING_RADIUS),
            opp[2] / MAX_SPEED, opp[3] / MAX_SPEED,
            (RING_OUT_R - own_dist) / RING_RADIUS,
            (RING_OUT_R - opp_dist) / RING_RADIUS,
            angle_to_opp / math.pi, angle_to_center / math.pi,
        ], dtype=np.float32)

    @staticmethod
    def _random_policy(obs):
        return np.random.randint(0, 9)

    def get_state(self):
        return {
            "red":  {"x": self.agents[0][0], "z": self.agents[0][1],
                     "vx": self.agents[0][2], "vz": self.agents[0][3]},
            "blue": {"x": self.agents[1][0], "z": self.agents[1][1],
                     "vx": self.agents[1][2], "vz": self.agents[1][3]},
        }
