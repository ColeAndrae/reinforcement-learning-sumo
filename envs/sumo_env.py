"""
SumoEnv v4 — Egocentric observations, no side-swapping.

KEY CHANGE: Observations are fully relative/egocentric.
No absolute positions means the policy works identically
regardless of which side the agent is on. No mirroring needed.

OBSERVATION (10 dims):
  [0]  dist_to_opp / (2*R)           how far is opponent
  [1]  angle_to_opp / pi             direction to opponent (-1 to 1)
  [2]  my_speed / MAX_V              how fast am I going
  [3]  my_heading / pi               direction I'm moving (-1 to 1)
  [4]  opp_speed / MAX_V             how fast is opponent going
  [5]  opp_heading / pi              direction opponent is moving
  [6]  my_edge_dist / R              how far from ring edge am I
  [7]  opp_edge_dist / R             how far from ring edge is opp
  [8]  closing_speed / MAX_V         rate of distance change (+ = closing)
  [9]  opp_angle_from_center / pi    which direction is opp from center

ACTIONS (9 discrete):
  0=noop  1=+x  2=-x  3=+z  4=-z  5=+x+z  6=+x-z  7=-x+z  8=-x-z
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

RING_RADIUS = 5.5
RING_OUT_R  = 5.3
CUBE_SIZE   = 0.9
PUSH_FORCE  = 28.0
DIAG_FORCE  = PUSH_FORCE / math.sqrt(2)
FRICTION    = 0.92
BOUNCE      = 0.3
DT          = 1.0 / 60.0
MAX_SPEED   = 15.0
MAX_STEPS   = 600
SUBSTEPS    = 2
OBS_DIM     = 10

ACTION_FORCES = [
    (0, 0), (PUSH_FORCE, 0), (-PUSH_FORCE, 0),
    (0, PUSH_FORCE), (0, -PUSH_FORCE),
    (DIAG_FORCE, DIAG_FORCE), (DIAG_FORCE, -DIAG_FORCE),
    (-DIAG_FORCE, DIAG_FORCE), (-DIAG_FORCE, -DIAG_FORCE),
]

def _len(x, z):
    return math.sqrt(x*x + z*z) or 1e-8


class SumoEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent_policy=None):
        super().__init__()
        self.opponent_policy = opponent_policy or (lambda obs: np.random.randint(0, 9))
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(-2, 2, shape=(OBS_DIM,), dtype=np.float32)
        self.state = [np.zeros(4) for _ in range(2)]  # [x, z, vx, vz]
        self.steps = 0
        self._prev_dist = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        j = self.np_random.uniform(-0.3, 0.3, 2)
        self.state[0] = np.array([-2.5+j[0], j[1], 0., 0.])
        self.state[1] = np.array([ 2.5-j[0],-j[1], 0., 0.])
        self.steps = 0
        self._prev_dist = _len(self.state[1][0]-self.state[0][0],
                               self.state[1][1]-self.state[0][1])
        return self._obs(0), {}

    def step(self, action):
        a0 = int(action)
        a1 = int(self.opponent_policy(self._obs(1)))

        terminated = False
        winner = -1

        for _ in range(SUBSTEPS):
            for i, a in enumerate([a0, a1]):
                fx, fz = ACTION_FORCES[a]
                self.state[i][2] += fx * DT
                self.state[i][3] += fz * DT
            f = FRICTION ** (DT * 60)
            for i in range(2):
                self.state[i][2] *= f
                self.state[i][3] *= f
            self._collide()
            for i in range(2):
                self.state[i][0] += self.state[i][2] * DT
                self.state[i][1] += self.state[i][3] * DT
            for i in range(2):
                if _len(self.state[i][0], self.state[i][1]) > RING_OUT_R:
                    terminated = True
                    winner = 1 - i
                    break
            if terminated:
                break

        self.steps += 1

        # Reward
        me, opp = self.state[0], self.state[1]
        dx, dz = opp[0]-me[0], opp[1]-me[1]
        dist = _len(dx, dz)
        my_dc = _len(me[0], me[1])
        opp_dc = _len(opp[0], opp[1])

        if terminated:
            reward = 10.0 if winner == 0 else -10.0
        else:
            reward = 0.0
            # Delta distance: reward for closing in
            reward += (self._prev_dist - dist) * 0.3
            # Opponent edge pressure
            reward += (opp_dc / RING_RADIUS) * 0.01
            # Self preservation
            if my_dc > RING_OUT_R * 0.7:
                reward -= 0.04 * (my_dc / RING_OUT_R)
            # Engagement bonus
            if dist < 2.5:
                reward += 0.015
            # Time penalty
            reward -= 0.001

        self._prev_dist = dist

        truncated = self.steps >= MAX_STEPS
        if truncated and not terminated:
            reward -= 1.5 if opp_dc < my_dc else 0.5

        return self._obs(0), reward, terminated, truncated, {"winner": winner}

    def _collide(self):
        a, b = self.state
        dx, dz = b[0]-a[0], b[1]-a[1]
        if abs(dx) < CUBE_SIZE and abs(dz) < CUBE_SIZE:
            d = _len(dx, dz)
            nx, nz = dx/d, dz/d
            ol = CUBE_SIZE - d
            if ol > 0:
                a[0] -= nx*ol*0.5; a[1] -= nz*ol*0.5
                b[0] += nx*ol*0.5; b[1] += nz*ol*0.5
            rv = (a[2]-b[2])*nx + (a[3]-b[3])*nz
            if rv > 0:
                imp = rv * (1 + BOUNCE)
                a[2] -= imp*nx*0.5; a[3] -= imp*nz*0.5
                b[2] += imp*nx*0.5; b[3] += imp*nz*0.5

    def _obs(self, idx):
        """Egocentric observation — identical math must be replicated in JS."""
        me = self.state[idx]
        opp = self.state[1-idx]

        dx, dz = opp[0]-me[0], opp[1]-me[1]
        dist = _len(dx, dz)
        angle_to_opp = math.atan2(dz, dx)

        my_speed = _len(me[2], me[3])
        my_heading = math.atan2(me[3], me[2]) if my_speed > 0.01 else 0.0

        opp_speed = _len(opp[2], opp[3])
        opp_heading = math.atan2(opp[3], opp[2]) if opp_speed > 0.01 else 0.0

        my_edge = RING_OUT_R - _len(me[0], me[1])
        opp_edge = RING_OUT_R - _len(opp[0], opp[1])

        # Closing speed: positive means getting closer
        if dist > 0.01:
            rel_vx = me[2] - opp[2]
            rel_vz = me[3] - opp[3]
            closing = (rel_vx * dx + rel_vz * dz) / dist
        else:
            closing = 0.0

        opp_angle_from_center = math.atan2(opp[1], opp[0])

        return np.array([
            dist / (2 * RING_RADIUS),
            angle_to_opp / math.pi,
            my_speed / MAX_SPEED,
            my_heading / math.pi,
            opp_speed / MAX_SPEED,
            opp_heading / math.pi,
            my_edge / RING_RADIUS,
            opp_edge / RING_RADIUS,
            closing / MAX_SPEED,
            opp_angle_from_center / math.pi,
        ], dtype=np.float32)


# ── Heuristic opponents (use egocentric obs) ────────────────

def heuristic_aggressive(obs):
    """Charges at opponent, avoids edge."""
    angle_to_opp = obs[1] * math.pi
    my_edge = obs[6] * RING_RADIUS
    dist = obs[0] * 2 * RING_RADIUS

    if my_edge < 1.2:
        # Near edge: move toward center (opposite of our position from center)
        # Use opp_angle_from_center as rough proxy for "away from edge"
        return _angle_to_action(angle_to_opp + math.pi)  # back off first
    return _angle_to_action(angle_to_opp)


def heuristic_flanker(obs):
    """Circles then charges."""
    angle_to_opp = obs[1] * math.pi
    dist = obs[0] * 2 * RING_RADIUS
    my_edge = obs[6] * RING_RADIUS

    if my_edge < 1.0:
        return _angle_to_action(angle_to_opp + math.pi)
    if dist > 2.5:
        return _angle_to_action(angle_to_opp + 0.6)  # circle
    return _angle_to_action(angle_to_opp)  # charge


def heuristic_positional(obs):
    """Stays near center, charges when close."""
    angle_to_opp = obs[1] * math.pi
    dist = obs[0] * 2 * RING_RADIUS
    my_edge = obs[6] * RING_RADIUS

    if my_edge < 2.0:
        return _angle_to_action(angle_to_opp + math.pi * 0.5)
    if dist < 3.0:
        return _angle_to_action(angle_to_opp)
    return _angle_to_action(angle_to_opp * 0.5)  # drift toward opponent slowly


def _angle_to_action(angle):
    """Convert angle (radians) to nearest discrete action."""
    angle = angle % (2 * math.pi)
    sector = round(angle / (math.pi / 4)) % 8
    return {0:1, 1:5, 2:3, 3:7, 4:2, 5:8, 6:4, 7:6}.get(sector, 0)
