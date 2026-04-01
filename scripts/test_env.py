#!/usr/bin/env python3
"""Quick smoke test for the sumo environment."""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.sumo_env import SumoEnv
from envs.self_play_env import SelfPlaySumoEnv


def test_basic_env():
    print("Testing SumoEnv...")
    env = SumoEnv()
    obs, info = env.reset()
    assert obs.shape == (12,), f"Obs shape: {obs.shape}"
    print(f"  Obs shape: {obs.shape}")
    print(f"  Obs range: [{obs.min():.2f}, {obs.max():.2f}]")

    # Run a few random episodes
    wins, losses, draws = 0, 0, 0
    steps_list = []
    for ep in range(100):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        steps_list.append(steps)
        if info.get("winner") == 0:
            wins += 1
        elif info.get("winner") == 1:
            losses += 1
        else:
            draws += 1

    avg_steps = np.mean(steps_list)
    print(f"  100 random episodes: {wins}W/{losses}L/{draws}D, avg {avg_steps:.0f} steps")
    print("  ✓ SumoEnv works!\n")


def test_self_play_env():
    print("Testing SelfPlaySumoEnv...")
    env = SelfPlaySumoEnv()
    obs, info = env.reset()
    assert obs.shape == (12,), f"Obs shape: {obs.shape}"

    # Run a few episodes
    for ep in range(20):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    print("  ✓ SelfPlaySumoEnv works!\n")


def test_performance():
    print("Performance benchmark...")
    env = SumoEnv()
    obs, _ = env.reset()

    start = time.time()
    total_steps = 0
    for _ in range(100):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            total_steps += 1

    elapsed = time.time() - start
    fps = total_steps / elapsed
    print(f"  {total_steps:,} steps in {elapsed:.2f}s = {fps:,.0f} steps/sec")
    print(f"  At PPO's 2048 steps/batch, each batch takes ~{2048/fps:.2f}s")
    print(f"  Estimated 500K steps: ~{500_000 / fps / 60:.1f} minutes")
    print("  ✓ Performance OK!\n")


if __name__ == "__main__":
    test_basic_env()
    test_self_play_env()
    test_performance()
    print("All tests passed! Ready to train.")
