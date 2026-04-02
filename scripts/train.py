#!/usr/bin/env python3
"""
train.py v3 — Curriculum self-play training.

Phase 1 (0-30%):  vs heuristic opponents (learn basics)
Phase 2 (30-60%): vs heuristics + self-play pool
Phase 3 (60-100%): mostly self-play (refine strategy)

Default: 2M steps, ~10-15 min on M2.
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from envs.self_play_env import SelfPlaySumoEnv
from envs.sumo_env import (
    SumoEnv, heuristic_aggressive, heuristic_flanker,
    heuristic_positional, random_policy, OBS_DIM
)

DEFAULT_TIMESTEPS = 2_000_000
CHECKPOINT_INTERVAL = 100_000


class CurriculumCallback(BaseCallback):
    def __init__(self, env, save_dir, total_timesteps, checkpoint_interval=CHECKPOINT_INTERVAL):
        super().__init__(verbose=1)
        self.env = env
        self.save_dir = save_dir
        self.total_timesteps = total_timesteps
        self.checkpoint_interval = checkpoint_interval
        self.next_checkpoint = checkpoint_interval
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self):
        # Update curriculum progress
        progress = self.num_timesteps / self.total_timesteps
        self.env.set_training_progress(progress)

        # Track outcomes
        for info in self.locals.get("infos", []):
            if "winner" in info and info["winner"] >= 0:
                if info["winner"] == 0:
                    self.wins += 1
                else:
                    self.losses += 1
            elif any(self.locals.get("dones", [False])):
                self.draws += 1

        # Checkpoint
        if self.num_timesteps >= self.next_checkpoint:
            self.next_checkpoint += self.checkpoint_interval

            ckpt = os.path.join(self.save_dir, f"ckpt_{self.num_timesteps}")
            self.model.save(ckpt)
            frozen = PPO.load(ckpt)
            self.env.update_opponent(frozen)

            total = self.wins + self.losses + self.draws
            wr = self.wins / max(1, total)
            elapsed = time.time() - self.start_time
            phase = "HEURISTIC" if progress < 0.3 else ("MIXED" if progress < 0.6 else "SELF-PLAY")

            print(
                f"  {self.num_timesteps:>10,} | {phase:>10} | "
                f"WR: {wr:.0%} ({self.wins}W/{self.losses}L/{self.draws}D) | "
                f"Pool: {len(self.env._opponent_pool)} | "
                f"{elapsed:.0f}s"
            )
            self.wins = self.losses = self.draws = 0

        return True


def evaluate_vs(model, opponent_fn, name, n_episodes=100):
    """Evaluate trained model against a specific opponent."""
    env = SumoEnv(opponent_policy=opponent_fn)
    wins = 0
    total_steps = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_steps += 1
        if info.get("winner") == 0:
            wins += 1
    wr = wins / n_episodes
    avg_len = total_steps / n_episodes
    print(f"  vs {name:>15}: {wr:>5.0%} win rate ({wins}/{n_episodes}), avg {avg_len:.0f} steps")
    return wr


def validate_obs_consistency(model):
    """Verify observations produce consistent actions — catches coordinate bugs."""
    env = SumoEnv()
    print("\n  Observation validation:")

    # Test: agent at center facing opponent at +x should predict move toward +x
    env.reset()
    env.agents[0] = np.array([0.0, 0.0, 0.0, 0.0])
    env.agents[1] = np.array([3.0, 0.0, 0.0, 0.0])
    obs = env._get_obs(0)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    # Actions that move +x: 1, 5, 6
    moves_toward = action in [1, 5, 6]
    print(f"    Opp at +x → action {action} ({'TOWARD' if moves_toward else 'AWAY'} ✓)" if moves_toward else
          f"    Opp at +x → action {action} (not toward, may indicate issue)")

    # Test: agent at center facing opponent at -x
    env.agents[0] = np.array([0.0, 0.0, 0.0, 0.0])
    env.agents[1] = np.array([-3.0, 0.0, 0.0, 0.0])
    obs = env._get_obs(0)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    moves_toward = action in [2, 7, 8]
    print(f"    Opp at -x → action {action} ({'TOWARD' if moves_toward else 'AWAY'} ✓)" if moves_toward else
          f"    Opp at -x → action {action} (not toward, may indicate issue)")

    # Test: agent near edge should move toward center
    env.agents[0] = np.array([4.5, 0.0, 0.0, 0.0])
    env.agents[1] = np.array([-2.0, 0.0, 0.0, 0.0])
    obs = env._get_obs(0)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    moves_center = action in [2, 7, 8]  # -x direction
    print(f"    At +x edge → action {action} ({'TO CENTER' if moves_center else 'RISKY'})")


def main():
    parser = argparse.ArgumentParser(description="Train Sumo Cube v3")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    args = parser.parse_args()

    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 65)
    print("  CUBE SUMO v3 — Curriculum Self-Play Training")
    print("=" * 65)
    print(f"  Timesteps:  {args.timesteps:,}")
    print(f"  Obs dims:   {OBS_DIM}")
    print(f"  Actions:    9 (cardinal + diagonal)")
    print(f"  Phases:     HEURISTIC → MIXED → SELF-PLAY")
    print(f"  Network:    256 → 256 (policy)  |  256 → 128 (value)")
    print("=" * 65)

    env = SelfPlaySumoEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 128],
            ),
        ),
    )

    callback = CurriculumCallback(
        env=env,
        save_dir=model_dir,
        total_timesteps=args.timesteps,
        checkpoint_interval=args.checkpoint_interval,
    )

    print("\nTraining...\n")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    final_path = os.path.join(model_dir, "sumo_final")
    model.save(final_path)
    print(f"\nModel saved to {final_path}")

    # ── Evaluation ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EVALUATION")
    print("=" * 65)

    results = {}
    results['random'] = evaluate_vs(model, random_policy, "Random")
    results['aggressive'] = evaluate_vs(model, heuristic_aggressive, "Aggressive")
    results['flanker'] = evaluate_vs(model, heuristic_flanker, "Flanker")
    results['positional'] = evaluate_vs(model, heuristic_positional, "Positional")

    validate_obs_consistency(model)

    min_wr = min(results.values())
    if min_wr < 0.6:
        print(f"\n  ⚠ Lowest win rate is {min_wr:.0%} — consider training longer")
    else:
        print(f"\n  ✓ All win rates ≥60%. Agent looks solid!")

    print(f"\n  Next: python3 scripts/export_to_js.py && python3 build_arena.py")


if __name__ == "__main__":
    main()
