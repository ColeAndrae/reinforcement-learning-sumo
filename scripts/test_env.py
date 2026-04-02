#!/usr/bin/env python3
"""Quick smoke test for v3 sumo environment."""

import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.sumo_env import SumoEnv, heuristic_aggressive, heuristic_flanker, heuristic_positional, random_policy, OBS_DIM
from envs.self_play_env import SelfPlaySumoEnv


def test_env():
    print("Testing SumoEnv v3...")
    env = SumoEnv()
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"
    assert env.action_space.n == 9
    print(f"  Obs: {OBS_DIM} dims, Actions: 9")
    print(f"  Obs range: [{obs.min():.2f}, {obs.max():.2f}]")

    for name, fn in [("Random", random_policy), ("Aggressive", heuristic_aggressive),
                     ("Flanker", heuristic_flanker), ("Positional", heuristic_positional)]:
        env2 = SumoEnv(opponent_policy=fn)
        w, l = 0, 0
        for _ in range(50):
            obs, _ = env2.reset()
            done = False
            while not done:
                obs, _, term, trunc, info = env2.step(env2.action_space.sample())
                done = term or trunc
            if info.get("winner") == 0: w += 1
            elif info.get("winner") == 1: l += 1
        print(f"  Random vs {name:>12}: {w}W/{l}L (of 50)")
    print("  ✓ SumoEnv OK\n")


def test_self_play():
    print("Testing SelfPlaySumoEnv...")
    env = SelfPlaySumoEnv()
    for _ in range(20):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(env.action_space.sample())
            done = term or trunc
    print("  ✓ SelfPlaySumoEnv OK\n")


def test_perf():
    print("Performance...")
    env = SumoEnv()
    start = time.time()
    steps = 0
    for _ in range(200):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            steps += 1
    elapsed = time.time() - start
    fps = steps / elapsed
    print(f"  {fps:,.0f} steps/sec → 2M steps ≈ {2_000_000/fps/60:.1f} min")
    print("  ✓ Performance OK\n")


if __name__ == "__main__":
    test_env()
    test_self_play()
    test_perf()
    print("All tests passed.")
