#!/usr/bin/env python3
"""
train.py — Train sumo cube agents via PPO self-play.
v2: 9 actions, stronger rewards, 2M steps default.
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

DEFAULT_TIMESTEPS = 2_000_000
CHECKPOINT_INTERVAL = 100_000


class SelfPlayCallback(BaseCallback):
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

        if self.num_timesteps >= self.next_checkpoint:
            self.next_checkpoint += self.checkpoint_interval

            ckpt_path = os.path.join(self.save_dir, f"ckpt_{self.num_timesteps}")
            self.model.save(ckpt_path)

            frozen = PPO.load(ckpt_path)
            self.env.update_opponent(frozen)

            elapsed = time.time() - self.start_time
            total_games = self.wins + self.losses + self.draws
            win_rate = self.wins / max(1, total_games)
            fps = self.num_timesteps / max(0.1, elapsed)

            if self.verbose:
                print(
                    f"  Step {self.num_timesteps:>10,} | "
                    f"Win rate: {win_rate:.1%} ({self.wins}W/{self.losses}L/{self.draws}D) | "
                    f"Pool: {len(self.env._opponent_pool)} | "
                    f"FPS: {fps:.0f} | "
                    f"Time: {elapsed:.0f}s"
                )

            self.wins = 0
            self.losses = 0
            self.draws = 0

        return True


def evaluate(model, n_episodes=100):
    env = SelfPlaySumoEnv()
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        if info.get("winner") == 0:
            wins += 1
    return wins / n_episodes


def main():
    parser = argparse.ArgumentParser(description="Train Sumo Cube agents v2")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--eval-only", type=str, default=None)
    args = parser.parse_args()

    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)

    if args.eval_only:
        print(f"Evaluating {args.eval_only} ...")
        model = PPO.load(args.eval_only)
        win_rate = evaluate(model, n_episodes=200)
        print(f"Win rate vs random: {win_rate:.1%}")
        return

    print("=" * 60)
    print("  CUBE SUMO v2 — Self-Play Training")
    print("  9 actions | stronger rewards | delta-based shaping")
    print("=" * 60)
    print(f"  Timesteps:  {args.timesteps:,}")
    print(f"  Checkpoints every: {args.checkpoint_interval:,} steps")
    print(f"  Save dir:   {model_dir}")
    print("=" * 60)

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
                pi=[256, 128],
                vf=[256, 128],
            ),
        ),
    )

    callback = SelfPlayCallback(
        env=env,
        save_dir=model_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    print("\nTraining started...\n")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    final_path = os.path.join(model_dir, "sumo_final")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    print("\nEvaluating vs random opponent (200 games)...")
    win_rate = evaluate(model, n_episodes=200)
    print(f"Win rate vs random: {win_rate:.1%}")

    print("\nEvaluating vs self (200 games, expect ~50%)...")
    env2 = SelfPlaySumoEnv()
    env2.set_opponent_from_model(model)
    wins_self = 0
    for _ in range(200):
        obs, _ = env2.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env2.step(int(action))
            done = terminated or truncated
        if info.get("winner") == 0:
            wins_self += 1
    print(f"Win rate vs self: {wins_self / 200:.1%}")

    print("\n✓ Training complete! Run: python3 scripts/export_to_js.py && python3 build_arena.py")


if __name__ == "__main__":
    main()
