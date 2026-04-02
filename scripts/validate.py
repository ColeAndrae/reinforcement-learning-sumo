#!/usr/bin/env python3
"""
validate.py — Simulate the browser scenario in Python.

Runs the trained agent through the EXACT same code path the browser uses:
  1. Two cubes at starting positions
  2. Build egocentric obs
  3. Forward pass through exported JSON weights
  4. Apply action, step physics
  5. Print what happens

If the agent runs off screen here, it'll run off screen in the browser.
"""

import os, sys, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Physics constants (must match browser)
R = 5.5
ROUT = 5.3
CS = 0.9
PF = 28.0
DF = PF / math.sqrt(2)
FR = 0.92
BN = 0.3
DT = 1/60
MV = 15.0
SS = 2

AF = [(0,0),(PF,0),(-PF,0),(0,PF),(0,-PF),(DF,DF),(DF,-DF),(-DF,DF),(-DF,-DF)]
ACTION_NAMES = ['NOOP','+X','-X','+Z','-Z','+X+Z','+X-Z','-X+Z','-X-Z']


def _len(x, z):
    return math.sqrt(x*x + z*z) or 1e-8


def build_obs(me_x, me_z, me_vx, me_vz, opp_x, opp_z, opp_vx, opp_vz):
    """EXACT copy of JS buildObs function."""
    dx = opp_x - me_x
    dz = opp_z - me_z
    dist = _len(dx, dz)
    angle_to_opp = math.atan2(dz, dx)

    my_speed = _len(me_vx, me_vz)
    my_heading = math.atan2(me_vz, me_vx) if my_speed > 0.01 else 0.0

    opp_speed = _len(opp_vx, opp_vz)
    opp_heading = math.atan2(opp_vz, opp_vx) if opp_speed > 0.01 else 0.0

    my_edge = ROUT - _len(me_x, me_z)
    opp_edge = ROUT - _len(opp_x, opp_z)

    if dist > 0.01:
        closing = ((me_vx - opp_vx) * dx + (me_vz - opp_vz) * dz) / dist
    else:
        closing = 0.0

    opp_angle_center = math.atan2(opp_z, opp_x)

    return [
        dist / (2*R),
        angle_to_opp / math.pi,
        my_speed / MV,
        my_heading / math.pi,
        opp_speed / MV,
        opp_heading / math.pi,
        my_edge / R,
        opp_edge / R,
        closing / MV,
        opp_angle_center / math.pi,
    ]


def forward_pass(policy, obs):
    """EXACT copy of JS aiPredict."""
    x = np.array(obs, dtype=np.float64)
    for i, layer in enumerate(policy['layers']):
        w = np.array(layer['weight'])
        b = np.array(layer['bias'])
        x = x @ w.T + b
        if i < len(policy['layers']) - 1:
            x = np.tanh(x)
    return int(np.argmax(x))


def simulate_match(policy, n_steps=300, verbose=True):
    """Simulate a full match using exported JSON weights."""
    # Initial positions (same as browser)
    rx, rz, rvx, rvz = -2.5, 0, 0, 0
    bx, bz, bvx, bvz =  2.5, 0, 0, 0

    for step in range(n_steps):
        # Build observations (same as browser)
        obs_r = build_obs(rx, rz, rvx, rvz, bx, bz, bvx, bvz)
        obs_b = build_obs(bx, bz, bvx, bvz, rx, rz, rvx, rvz)

        # AI prediction (same as browser)
        act_r = forward_pass(policy, obs_r)
        act_b = forward_pass(policy, obs_b)

        if verbose and step < 20:
            r_dist = _len(rx, rz)
            b_dist = _len(bx, bz)
            print(f"  Step {step:>3}: R({rx:+.2f},{rz:+.2f}) → {ACTION_NAMES[act_r]:>5}  |  "
                  f"B({bx:+.2f},{bz:+.2f}) → {ACTION_NAMES[act_b]:>5}  |  "
                  f"R_edge:{ROUT-r_dist:.2f}  B_edge:{ROUT-b_dist:.2f}")

        # Physics (same as browser)
        for _ in range(SS):
            fx_r, fz_r = AF[act_r]
            fx_b, fz_b = AF[act_b]
            rvx += fx_r * DT; rvz += fz_r * DT
            bvx += fx_b * DT; bvz += fz_b * DT
            f = FR ** (DT * 60)
            rvx *= f; rvz *= f; bvx *= f; bvz *= f

            # Collision
            dx, dz = bx-rx, bz-rz
            if abs(dx) < CS and abs(dz) < CS:
                d = _len(dx, dz)
                nx, nz = dx/d, dz/d
                ol = CS - d
                if ol > 0:
                    rx -= nx*ol*0.5; rz -= nz*ol*0.5
                    bx += nx*ol*0.5; bz += nz*ol*0.5
                rv = (rvx-bvx)*nx + (rvz-bvz)*nz
                if rv > 0:
                    imp = rv*(1+BN)
                    rvx -= imp*nx*0.5; rvz -= imp*nz*0.5
                    bvx += imp*nx*0.5; bvz += imp*nz*0.5

            rx += rvx*DT; rz += rvz*DT
            bx += bvx*DT; bz += bvz*DT

            # Ring out
            if _len(rx, rz) > ROUT:
                if verbose: print(f"  >>> RED RING OUT at step {step}!")
                return 'blue', step
            if _len(bx, bz) > ROUT:
                if verbose: print(f"  >>> BLUE RING OUT at step {step}!")
                return 'red', step

    if verbose: print(f"  >>> DRAW (timeout)")
    return 'draw', n_steps


def main():
    policy_path = os.path.join(os.getcwd(), 'models', 'policy.json')
    if not os.path.exists(policy_path):
        print(f"No policy found at {policy_path}")
        print("Run: python3 scripts/export_to_js.py first")
        return

    with open(policy_path) as f:
        policy = json.load(f)

    print(f"Policy: {len(policy['layers'])} layers")
    for i, l in enumerate(policy['layers']):
        w = np.array(l['weight'])
        print(f"  Layer {i}: {w.shape[1]} → {w.shape[0]}")

    # Match the Python env obs against our build_obs
    from envs.sumo_env import SumoEnv
    env = SumoEnv()
    env.reset(seed=42)
    env.state[0] = np.array([-2.5, 0, 0, 0.])
    env.state[1] = np.array([ 2.5, 0, 0, 0.])

    py_obs = env._obs(0)
    js_obs = build_obs(-2.5, 0, 0, 0, 2.5, 0, 0, 0)

    print(f"\nObservation match test (start position):")
    match = True
    for i, (p, j) in enumerate(zip(py_obs, js_obs)):
        ok = abs(p - j) < 1e-6
        if not ok: match = False
        print(f"  [{i:>2}] Python={p:+.6f}  JS={j:+.6f}  {'✓' if ok else '✗ MISMATCH'}")
    print(f"  {'✓ All match!' if match else '✗ MISMATCH DETECTED — THIS IS THE BUG'}")

    # Simulate a match
    print(f"\nSimulating AI vs AI match (first 20 steps):")
    winner, steps = simulate_match(policy, n_steps=300, verbose=True)
    print(f"\nResult: {winner} wins after {steps} steps")

    # Run 50 matches silently
    print(f"\nRunning 50 matches...")
    results = {'red': 0, 'blue': 0, 'draw': 0}
    total_steps = 0
    for _ in range(50):
        w, s = simulate_match(policy, n_steps=600, verbose=False)
        results[w] += 1
        total_steps += s
    print(f"  Red: {results['red']}  Blue: {results['blue']}  Draw: {results['draw']}")
    print(f"  Avg match length: {total_steps/50:.0f} steps")

    if results['draw'] > 40:
        print("  ⚠ Mostly draws — agents may not be engaging")
    elif results['red'] > 45 or results['blue'] > 45:
        print("  ⚠ Very one-sided — possible asymmetry bug")
    else:
        print("  ✓ Balanced matches — looks good!")


if __name__ == '__main__':
    main()
