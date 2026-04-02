"""
SumoEnv v3 — Clean rewrite with verified coordinate system.

COORDINATE SYSTEM (matches Three.js browser exactly):
  - x: left/right   (red starts at x=-2.5, blue at x=+2.5)
  - z: forward/back  (Three.js z-axis, NOT y)
  - Ring is centered at origin, radius 5.5
  - Ring-out when sqrt(x^2 + z^2) > 5.3

STATE per agent: [x, z, vx, vz]  (indices 0,1,2,3)

OBSERVATION (14 dims — richer than v2):
  [0]  own_x / R                      normalized position
  [1]  own_z / R
  [2]  own_vx / MAX_V                 normalized velocity
  [3]  own_vz / MAX_V
  [4]  opp_x / R                      opponent absolute position
  [5]  opp_z / R
  [6]  opp_vx / MAX_V                 opponent velocity
  [7]  opp_vz / MAX_V
  [8]  dist_between / (2*R)           distance to opponent
  [9]  own_dist_center / R            own distance from center
  [10] opp_dist_center / R            opponent distance from center
  [11] dot_vel_toward_opp             how much own velocity points at opp (-1 to 1)
  [12] dot_opp_vel_toward_me          how much opp velocity points at me (-1 to 1)
  [13] cross_product_sign             which side opponent is on (-1 to 1)

ACTIONS (9 discrete):
  0=noop  1=+x  2=-x  3=+z  4=-z  5=+x+z  6=+x-z  7=-x+z  8=-x-z
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

# ── Constants (MUST match browser) ──────────────────────────
RING_RADIUS   = 5.5
RING_OUT_R    = RING_RADIUS - 0.2   # 5.3
CUBE_SIZE     = 0.9
PUSH_FORCE    = 28.0
DIAG_FORCE    = PUSH_FORCE / math.sqrt(2)
FRICTION      = 0.92
BOUNCE        = 0.3
FIXED_DT      = 1.0 / 60.0
MAX_SPEED     = 15.0
MAX_STEPS     = 600                  # 10 seconds at 60Hz
SUBSTEPS      = 2
OBS_DIM       = 14

ACTION_FORCES = [
    (0.0, 0.0),                       # 0: noop
    (PUSH_FORCE, 0.0),                # 1: +x
    (-PUSH_FORCE, 0.0),               # 2: -x
    (0.0, PUSH_FORCE),                # 3: +z
    (0.0, -PUSH_FORCE),               # 4: -z
    (DIAG_FORCE, DIAG_FORCE),         # 5: +x+z
    (DIAG_FORCE, -DIAG_FORCE),        # 6: +x-z
    (-DIAG_FORCE, DIAG_FORCE),        # 7: -x+z
    (-DIAG_FORCE, -DIAG_FORCE),       # 8: -x-z
]


def _dist(x, z):
    return math.sqrt(x*x + z*z) or 1e-8


def _normalize(x, z):
    d = _dist(x, z)
    return x/d, z/d


def _dot(ax, az, bx, bz):
    return ax*bx + az*bz


# ── Heuristic opponents ─────────────────────────────────────

def heuristic_aggressive(obs):
    """Charges straight at opponent, avoids edges."""
    own_x  = obs[0] * RING_RADIUS
    own_z  = obs[1] * RING_RADIUS
    opp_x  = obs[4] * RING_RADIUS
    opp_z  = obs[5] * RING_RADIUS
    own_dc = obs[9] * RING_RADIUS      # distance from center

    # Direction toward opponent
    dx = opp_x - own_x
    dz = opp_z - own_z

    # If near edge, bias toward center
    if own_dc > RING_RADIUS * 0.6:
        cx, cz = -own_x, -own_z  # toward center
        cn = _dist(cx, cz)
        if cn > 0:
            dx += cx / cn * 15
            dz += cz / cn * 15

    return _direction_to_action(dx, dz)


def heuristic_flanker(obs):
    """Circles around opponent, attacks from the side."""
    own_x  = obs[0] * RING_RADIUS
    own_z  = obs[1] * RING_RADIUS
    opp_x  = obs[4] * RING_RADIUS
    opp_z  = obs[5] * RING_RADIUS
    own_dc = obs[9] * RING_RADIUS

    dx = opp_x - own_x
    dz = opp_z - own_z
    d = _dist(dx, dz)

    if d > 2.5:
        # Circle: perpendicular + approach
        px, pz = -dz/d, dx/d
        dx = dx/d * 0.3 + px * 0.7
        dz = dz/d * 0.3 + pz * 0.7
    # else charge straight

    if own_dc > RING_RADIUS * 0.6:
        cx, cz = _normalize(-own_x, -own_z)
        dx += cx * 12
        dz += cz * 12

    return _direction_to_action(dx, dz)


def heuristic_positional(obs):
    """Tries to stay between opponent and center, then push out."""
    own_x  = obs[0] * RING_RADIUS
    own_z  = obs[1] * RING_RADIUS
    opp_x  = obs[4] * RING_RADIUS
    opp_z  = obs[5] * RING_RADIUS
    opp_dc = obs[10] * RING_RADIUS

    # Direction: toward position between opp and center
    ideal_x = opp_x * 0.5  # halfway between opp and center
    ideal_z = opp_z * 0.5
    dx = ideal_x - own_x
    dz = ideal_z - own_z

    d_to_ideal = _dist(dx, dz)
    if d_to_ideal < 1.5:
        # Close enough to ideal pos — now push toward opponent
        dx = opp_x - own_x
        dz = opp_z - own_z

    return _direction_to_action(dx, dz)


def random_policy(obs):
    return np.random.randint(0, 9)


def _direction_to_action(dx, dz):
    """Convert a desired direction vector into the closest discrete action."""
    if abs(dx) < 0.1 and abs(dz) < 0.1:
        return 0
    angle = math.atan2(dz, dx)  # radians
    # Map angle to 8 directions
    # +x=0, +x+z=π/4, +z=π/2, -x+z=3π/4, -x=π, -x-z=-3π/4, -z=-π/2, +x-z=-π/4
    sector = round(angle / (math.pi / 4)) % 8
    action_map = {0: 1, 1: 5, 2: 3, 3: 7, 4: 2, 5: 8, 6: 4, 7: 6}
    return action_map.get(sector, 0)


ALL_HEURISTICS = [heuristic_aggressive, heuristic_flanker, heuristic_positional, random_policy]


class SumoEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent_policy=None, render_mode=None):
        super().__init__()
        self.opponent_policy = opponent_policy or random_policy
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.steps = 0
        # State: [x, z, vx, vz] for each agent
        self.agents = [np.zeros(4, dtype=np.float64) for _ in range(2)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        jx = self.np_random.uniform(-0.3, 0.3)
        jz = self.np_random.uniform(-0.3, 0.3)
        self.agents[0] = np.array([-2.5 + jx,  jz, 0.0, 0.0])
        self.agents[1] = np.array([ 2.5 - jx, -jz, 0.0, 0.0])
        self.steps = 0
        return self._get_obs(0), {}

    def step(self, action):
        action = int(action)
        opp_obs = self._get_obs(1)
        opp_action = int(self.opponent_policy(opp_obs))
        actions = [action, opp_action]

        terminated = False
        winner = -1

        for _ in range(SUBSTEPS):
            # Apply forces
            for i in range(2):
                fx, fz = ACTION_FORCES[actions[i]]
                self.agents[i][2] += fx * FIXED_DT  # vx += force * dt
                self.agents[i][3] += fz * FIXED_DT  # vz += force * dt

            # Friction
            fric = FRICTION ** (FIXED_DT * 60)
            for i in range(2):
                self.agents[i][2] *= fric
                self.agents[i][3] *= fric

            # Collision
            self._resolve_collision()

            # Integrate positions
            for i in range(2):
                self.agents[i][0] += self.agents[i][2] * FIXED_DT
                self.agents[i][1] += self.agents[i][3] * FIXED_DT

            # Ring-out check
            for i in range(2):
                if _dist(self.agents[i][0], self.agents[i][1]) > RING_OUT_R:
                    terminated = True
                    winner = 1 - i
                    break
            if terminated:
                break

        self.steps += 1

        # ── Reward ──────────────────────────────────────────
        reward = self._compute_reward(terminated, winner)

        truncated = self.steps >= MAX_STEPS
        if truncated and not terminated:
            # Stalemate: penalize based on who's closer to edge
            own_dc = _dist(self.agents[0][0], self.agents[0][1])
            opp_dc = _dist(self.agents[1][0], self.agents[1][1])
            if opp_dc > own_dc:
                reward += 1.0   # we have better position
            else:
                reward -= 1.0

        return self._get_obs(0), reward, terminated, truncated, {"winner": winner, "steps": self.steps}

    def _compute_reward(self, terminated, winner):
        if terminated:
            return 10.0 if winner == 0 else -10.0

        me, opp = self.agents[0], self.agents[1]
        own_dc = _dist(me[0], me[1])
        opp_dc = _dist(opp[0], opp[1])

        dx = opp[0] - me[0]
        dz = opp[1] - me[1]
        dist_between = _dist(dx, dz)

        # How much our velocity points toward the opponent
        my_speed = _dist(me[2], me[3])
        if my_speed > 0.1 and dist_between > 0.1:
            nx, nz = dx / dist_between, dz / dist_between
            closing_dot = _dot(me[2], me[3], nx, nz) / my_speed
        else:
            closing_dot = 0.0

        reward = 0.0

        # 1. Reward moving toward opponent (encourages engagement)
        reward += closing_dot * 0.03

        # 2. Reward opponent being far from center (edge pressure)
        reward += (opp_dc / RING_RADIUS) * 0.02

        # 3. Reward self being close to center (safety)
        reward -= (own_dc / RING_RADIUS) * 0.015

        # 4. Proximity bonus — reward being close enough to fight
        if dist_between < 3.0:
            reward += 0.01
        elif dist_between > 5.0:
            reward -= 0.01

        # 5. Strong danger zone penalty near edge
        if own_dc > RING_OUT_R * 0.75:
            reward -= 0.05 * (own_dc / RING_OUT_R)

        # 6. Tiny time penalty
        reward -= 0.001

        return reward

    def _resolve_collision(self):
        a, b = self.agents
        dx = b[0] - a[0]
        dz = b[1] - a[1]
        if abs(dx) < CUBE_SIZE and abs(dz) < CUBE_SIZE:
            dist = _dist(dx, dz)
            nx, nz = dx / dist, dz / dist
            overlap = CUBE_SIZE - dist
            if overlap > 0:
                a[0] -= nx * overlap * 0.5
                a[1] -= nz * overlap * 0.5
                b[0] += nx * overlap * 0.5
                b[1] += nz * overlap * 0.5
            rel_v = (a[2] - b[2]) * nx + (a[3] - b[3]) * nz
            if rel_v > 0:
                imp = rel_v * (1 + BOUNCE)
                a[2] -= imp * nx * 0.5
                a[3] -= imp * nz * 0.5
                b[2] += imp * nx * 0.5
                b[3] += imp * nz * 0.5

    def _get_obs(self, agent_idx):
        me = self.agents[agent_idx]
        opp = self.agents[1 - agent_idx]

        dx = opp[0] - me[0]
        dz = opp[1] - me[1]
        dist_between = _dist(dx, dz)
        own_dc = _dist(me[0], me[1])
        opp_dc = _dist(opp[0], opp[1])

        # Dot product of own velocity toward opponent
        my_speed = _dist(me[2], me[3])
        if my_speed > 0.01 and dist_between > 0.01:
            dot_vel = _dot(me[2], me[3], dx/dist_between, dz/dist_between) / max(my_speed, 0.01)
        else:
            dot_vel = 0.0

        # Dot product of opponent velocity toward me
        opp_speed = _dist(opp[2], opp[3])
        if opp_speed > 0.01 and dist_between > 0.01:
            dot_opp = _dot(opp[2], opp[3], -dx/dist_between, -dz/dist_between) / max(opp_speed, 0.01)
        else:
            dot_opp = 0.0

        # Cross product sign (which side is opponent on relative to my velocity)
        cross = me[2] * dz - me[3] * dx
        cross_sign = max(-1.0, min(1.0, cross / max(my_speed * dist_between, 0.01)))

        return np.array([
            me[0] / RING_RADIUS,
            me[1] / RING_RADIUS,
            me[2] / MAX_SPEED,
            me[3] / MAX_SPEED,
            opp[0] / RING_RADIUS,
            opp[1] / RING_RADIUS,
            opp[2] / MAX_SPEED,
            opp[3] / MAX_SPEED,
            dist_between / (2 * RING_RADIUS),
            own_dc / RING_RADIUS,
            opp_dc / RING_RADIUS,
            dot_vel,
            dot_opp,
            cross_sign,
        ], dtype=np.float32)
