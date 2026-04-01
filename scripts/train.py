#!/usr/bin/env python3
"""
train.py — Train sumo cube agents via PPO self-play.

This script:
  1. Creates a SelfPlaySumoEnv
  2. Trains a PPO agent for N total timesteps
  3. Every CHECKPOINT_INTERVAL steps, saves a checkpoint and
     adds the current policy to the opponent pool
  4. Exports the final model for browser use

Expected training time on Apple M2: ~15-30 minutes for solid play.

Usage:
    python train.py [--timesteps 500000] [--checkpoint-interval 25000]
"""

import os
import sys
import time
import argparse
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from envs.self_play_env import SelfPlaySumoEnv


# ── Config ──────────────────────────────────────────────────
DEFAULT_TIMESTEPS = 500_000
CHECKPOINT_INTERVAL = 25_000
MODEL_DIR = os.path.join(os.getcwd(), "models")


class SelfPlayCallback(BaseCallback):
    """
    Callback that:
      - Saves checkpoints at regular intervals
      - Updates the opponent pool from checkpoints
      - Logs win rate statistics
    """

    def __init__(self, env, save_dir, checkpoint_interval=CHECKPOINT_INTERVAL, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.save_dir = save_dir
        self.checkpoint_interval = checkpoint_interval
        self.next_checkpoint = checkpoint_interval
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.episode_count = 0
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self):
        # Track outcomes from info dicts
        for info in self.locals.get("infos", []):
            if "winner" in info and info["winner"] >= 0:
                self.episode_count += 1
                if info["winner"] == 0:
                    self.wins += 1
                else:
                    self.losses += 1
            elif self.locals.get("dones", [False])[0]:
                self.episode_count += 1
                self.draws += 1

        # Checkpoint & opponent update
        if self.num_timesteps >= self.next_checkpoint:
            self.next_checkpoint += self.checkpoint_interval

            # Save checkpoint
            ckpt_path = os.path.join(
                self.save_dir, f"ckpt_{self.num_timesteps}"
            )
            self.model.save(ckpt_path)

            # Load checkpoint as a frozen opponent
            frozen = PPO.load(ckpt_path)
            self.env.update_opponent(frozen)

            # Log stats
            elapsed = time.time() - self.start_time
            total_games = self.wins + self.losses + self.draws
            win_rate = self.wins / max(1, total_games)
            fps = self.num_timesteps / max(0.1, elapsed)

            if self.verbose:
                print(
                    f"  Step {self.num_timesteps:>8,} | "
                    f"Win rate: {win_rate:.1%} ({self.wins}W/{self.losses}L/{self.draws}D) | "
                    f"Pool size: {len(self.env._opponent_pool)} | "
                    f"FPS: {fps:.0f} | "
                    f"Time: {elapsed:.0f}s"
                )

            # Reset counters for next window
            self.wins = 0
            self.losses = 0
            self.draws = 0

        return True


def evaluate(model, n_episodes=100):
    """Quick evaluation: trained agent vs random."""
    env = SelfPlaySumoEnv()
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("winner") == 0:
            wins += 1
    return wins / n_episodes


def main():
    parser = argparse.ArgumentParser(description="Train Sumo Cube agents")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                        help="Total training timesteps")
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL,
                        help="Steps between checkpoints")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Path to model to evaluate (skip training)")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.eval_only:
        print(f"Evaluating {args.eval_only} ...")
        model = PPO.load(args.eval_only)
        win_rate = evaluate(model, n_episodes=200)
        print(f"Win rate vs random: {win_rate:.1%}")
        return

    print("=" * 60)
    print("  CUBE SUMO — Self-Play Training")
    print("=" * 60)
    print(f"  Timesteps:  {args.timesteps:,}")
    print(f"  Checkpoints every: {args.checkpoint_interval:,} steps")
    print(f"  Save dir:   {MODEL_DIR}")
    print("=" * 60)

    # Create environment
    env = SelfPlaySumoEnv()

    # PPO with tuned hyperparameters for this problem
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # entropy bonus for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128],   # policy network
                vf=[128, 128],   # value network
            ),
        ),
    )

    callback = SelfPlayCallback(
        env=env,
        save_dir=MODEL_DIR,
        checkpoint_interval=args.checkpoint_interval,
    )

    print("\nTraining started...\n")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
    )

    # Save final model
    final_path = os.path.join(MODEL_DIR, "sumo_final")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    # Evaluate
    print("\nEvaluating vs random opponent (200 games)...")
    win_rate = evaluate(model, n_episodes=200)
    print(f"Win rate vs random: {win_rate:.1%}")

    # Evaluate vs self
    print("Evaluating vs self (should be ~50%)...")
    env2 = SelfPlaySumoEnv()
    env2.set_opponent_from_model(model)
    wins_self = 0
    for _ in range(200):
        obs, _ = env2.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env2.step(action)
            done = terminated or truncated
        if info.get("winner") == 0:
            wins_self += 1
    print(f"Win rate vs self: {wins_self / 200:.1%}")

    print("\n✓ Training complete! Next: run export_to_js.py to deploy to browser.")


if __name__ == "__main__":
    main()
