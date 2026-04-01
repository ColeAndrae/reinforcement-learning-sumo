"""
SumoEnv — Gymnasium-compatible sumo wrestling environment.

Physics are matched to the Three.js browser arena:
  - Circular ring of radius 5.5
  - Cubes of size 0.9 with mass 1.0
  - Push force 28, friction 0.92, bounce 0.3
  - Ring-out when distance from center > RING_RADIUS - 0.2

Observation (12 dims per agent, relative frame):
  [0]  own_x / RING_RADIUS          (normalized position)
  [1]  own_z / RING_RADIUS
  [2]  own_vx / MAX_SPEED            (normalized velocity)
  [3]  own_vz / MAX_SPEED
  [4]  opp_rel_x / (2*RING_RADIUS)   (relative opponent position)
  [5]  opp_rel_z / (2*RING_RADIUS)
  [6]  opp_vx / MAX_SPEED            (opponent velocity)
  [7]  opp_vz / MAX_SPEED
  [8]  own_dist_to_edge / RING_RADIUS (how close to falling out)
  [9]  opp_dist_to_edge / RING_RADIUS
  [10] angle_to_opp / pi              (heading toward opponent)
  [11] angle_to_center / pi           (heading toward safety)

Actions (5 discrete):
  0 = no-op, 1 = +x, 2 = -x, 3 = +z, 4 = -z
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math


# ── Constants (matched to browser) ──────────────────────────
RING_RADIUS   = 5.5
RING_OUT_R    = RING_RADIUS - 0.2
CUBE_SIZE     = 0.9
CUBE_MASS     = 1.0
PUSH_FORCE    = 28.0
FRICTION      = 0.92
BOUNCE        = 0.3
FIXED_DT      = 1.0 / 60.0     # simulate at 60 Hz
MAX_SPEED     = 15.0            # for normalization
MAX_STEPS     = 600             # 10 seconds at 60 Hz
SUBSTEPS      = 2               # physics sub-steps per action


class SumoEnv(gym.Env):
    """
    Two-agent sumo environment.

    By default this env controls agent 0 (red).
    Agent 1 (blue) is driven by `opponent_policy` callback,
    which takes an observation and returns an action.

    For self-play training, use SelfPlaySumoEnv wrapper instead.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, opponent_policy=None, render_mode=None):
        super().__init__()

        self.opponent_policy = opponent_policy or self._random_policy

        # Action & observation spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(12,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.steps = 0

        # Agent states: [x, z, vx, vz]
        self.agents = [np.zeros(4, dtype=np.float64) for _ in range(2)]

    # ── Reset ───────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Spawn positions with slight randomness
        jitter = self.np_random.uniform(-0.3, 0.3, size=2)
        self.agents[0] = np.array([-2.5 + jitter[0], 0.0 + jitter[1], 0.0, 0.0])
        self.agents[1] = np.array([ 2.5 - jitter[0], 0.0 - jitter[1], 0.0, 0.0])
        self.steps = 0

        obs = self._get_obs(0)
        return obs, {}

    # ── Step ────────────────────────────────────────────────
    def step(self, action):
        """Step agent 0. Agent 1 acts via opponent_policy."""
        opp_obs = self._get_obs(1)
        opp_action = self.opponent_policy(opp_obs)

        actions = [int(action), int(opp_action)]

        # Sub-step physics for stability
        reward = 0.0
        terminated = False
        winner = -1

        for _ in range(SUBSTEPS):
            # Apply forces
            for i in range(2):
                fx, fz = self._action_to_force(actions[i])
                self.agents[i][2] += fx / CUBE_MASS * FIXED_DT
                self.agents[i][3] += fz / CUBE_MASS * FIXED_DT

            # Friction
            for i in range(2):
                f = FRICTION ** (FIXED_DT * 60)
                self.agents[i][2] *= f
                self.agents[i][3] *= f

            # Collision
            self._resolve_collision()

            # Integrate positions
            for i in range(2):
                self.agents[i][0] += self.agents[i][2] * FIXED_DT
                self.agents[i][1] += self.agents[i][3] * FIXED_DT

            # Ring-out check
            for i in range(2):
                dist = math.sqrt(self.agents[i][0]**2 + self.agents[i][1]**2)
                if dist > RING_OUT_R:
                    terminated = True
                    winner = 1 - i  # other agent wins
                    break

            if terminated:
                break

        self.steps += 1

        # ── Reward shaping ──────────────────────────────────
        if terminated:
            reward = 10.0 if winner == 0 else -10.0
        else:
            # Small reward for pushing opponent toward edge
            opp_dist = math.sqrt(self.agents[1][0]**2 + self.agents[1][1]**2)
            own_dist = math.sqrt(self.agents[0][0]**2 + self.agents[0][1]**2)
            reward += opp_dist * 0.005          # opponent near edge = good
            reward -= own_dist * 0.005          # self near edge = bad
            reward -= 0.001                     # time penalty (encourages action)

            # Small reward for closing distance (encourages engagement)
            dx = self.agents[1][0] - self.agents[0][0]
            dz = self.agents[1][1] - self.agents[0][1]
            dist_between = math.sqrt(dx*dx + dz*dz)
            reward -= dist_between * 0.001

        truncated = self.steps >= MAX_STEPS
        if truncated and not terminated:
            # Draw: slight penalty
            reward -= 1.0

        obs = self._get_obs(0)
        info = {"winner": winner, "steps": self.steps}

        return obs, reward, terminated, truncated, info

    # ── Physics helpers ─────────────────────────────────────
    def _action_to_force(self, action):
        """Map discrete action to force vector."""
        forces = {
            0: (0.0, 0.0),
            1: (PUSH_FORCE, 0.0),
            2: (-PUSH_FORCE, 0.0),
            3: (0.0, PUSH_FORCE),
            4: (0.0, -PUSH_FORCE),
        }
        return forces.get(action, (0.0, 0.0))

    def _resolve_collision(self):
        """AABB collision between the two cubes."""
        a, b = self.agents
        dx = b[0] - a[0]
        dz = b[1] - a[1]

        if abs(dx) < CUBE_SIZE and abs(dz) < CUBE_SIZE:
            dist = math.sqrt(dx*dx + dz*dz) or 0.01
            nx = dx / dist
            nz = dz / dist

            # Separate
            overlap = CUBE_SIZE - dist
            if overlap > 0:
                a[0] -= nx * overlap * 0.5
                a[1] -= nz * overlap * 0.5
                b[0] += nx * overlap * 0.5
                b[1] += nz * overlap * 0.5

            # Impulse
            rel_v = (a[2] - b[2]) * nx + (a[3] - b[3]) * nz
            if rel_v > 0:
                impulse = rel_v * (1 + BOUNCE)
                a[2] -= impulse * nx * 0.5
                a[3] -= impulse * nz * 0.5
                b[2] += impulse * nx * 0.5
                b[3] += impulse * nz * 0.5

    # ── Observations ────────────────────────────────────────
    def _get_obs(self, agent_idx):
        """Build observation vector for given agent."""
        me = self.agents[agent_idx]
        opp = self.agents[1 - agent_idx]

        own_dist = math.sqrt(me[0]**2 + me[1]**2)
        opp_dist = math.sqrt(opp[0]**2 + opp[1]**2)

        rel_x = opp[0] - me[0]
        rel_z = opp[1] - me[1]
        angle_to_opp = math.atan2(rel_z, rel_x)
        angle_to_center = math.atan2(-me[1], -me[0])

        obs = np.array([
            me[0] / RING_RADIUS,
            me[1] / RING_RADIUS,
            me[2] / MAX_SPEED,
            me[3] / MAX_SPEED,
            rel_x / (2 * RING_RADIUS),
            rel_z / (2 * RING_RADIUS),
            opp[2] / MAX_SPEED,
            opp[3] / MAX_SPEED,
            (RING_OUT_R - own_dist) / RING_RADIUS,
            (RING_OUT_R - opp_dist) / RING_RADIUS,
            angle_to_opp / math.pi,
            angle_to_center / math.pi,
        ], dtype=np.float32)

        return obs

    # ── Default opponent ────────────────────────────────────
    @staticmethod
    def _random_policy(obs):
        return np.random.randint(0, 5)

    # ── State access (for export / visualization) ───────────
    def get_state(self):
        """Return raw positions and velocities for both agents."""
        return {
            "red":  {"x": self.agents[0][0], "z": self.agents[0][1],
                     "vx": self.agents[0][2], "vz": self.agents[0][3]},
            "blue": {"x": self.agents[1][0], "z": self.agents[1][1],
                     "vx": self.agents[1][2], "vz": self.agents[1][3]},
        }
