#!/usr/bin/env python3
"""
export_to_js.py — Export a trained PPO policy to JSON for browser inference.

The policy is a simple MLP: obs(12) → 128 → 128 → action_logits(5)
We extract weights + biases and save as JSON so a tiny JS forward pass
can replicate the agent in the Three.js arena.

Usage:
    python export_to_js.py [--model ../models/sumo_final]
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO


def extract_policy_weights(model):
    """
    Extract the policy (actor) network weights from a PPO model.
    SB3's MlpPolicy stores the policy network in:
      model.policy.mlp_extractor.policy_net  (shared layers)
      model.policy.action_net                (final logits layer)
    """
    policy = model.policy

    layers = []

    # Shared MLP extractor — policy branch
    for name, param in policy.mlp_extractor.policy_net.named_parameters():
        layers.append({
            "name": f"policy_net.{name}",
            "shape": list(param.shape),
            "data": param.detach().cpu().numpy().tolist(),
        })

    # Final action layer
    for name, param in policy.action_net.named_parameters():
        layers.append({
            "name": f"action_net.{name}",
            "shape": list(param.shape),
            "data": param.detach().cpu().numpy().tolist(),
        })

    return layers


def extract_obs_normalization(model):
    """
    If the model uses VecNormalize, extract observation mean/var.
    Our env doesn't use VecNormalize, so this returns None.
    """
    return None


def build_js_model(layers):
    """
    Build a minimal JSON representation of the model:
    {
      "architecture": "mlp",
      "activation": "tanh",
      "layers": [
        {"weight": [[...]], "bias": [...]},
        ...
      ]
    }
    """
    # Group into (weight, bias) pairs
    processed = []
    i = 0
    while i < len(layers):
        w_layer = layers[i]
        b_layer = layers[i + 1] if i + 1 < len(layers) else None

        if "weight" in w_layer["name"] and b_layer and "bias" in b_layer["name"]:
            processed.append({
                "weight": w_layer["data"],
                "bias": b_layer["data"],
            })
            i += 2
        else:
            i += 1

    return {
        "architecture": "mlp",
        "activation": "tanh",
        "layers": processed,
    }


def verify_export(model, js_model):
    """Verify the exported model produces identical outputs."""
    # Run a few random observations through both
    env_obs = np.random.uniform(-1, 1, size=(10, 12)).astype(np.float32)
    errors = []

    for obs in env_obs:
        # SB3 prediction
        action_sb3, _ = model.predict(obs, deterministic=True)

        # Manual forward pass (replicating JS logic)
        x = obs.copy()
        for i, layer in enumerate(js_model["layers"]):
            w = np.array(layer["weight"])
            b = np.array(layer["bias"])
            x = x @ w.T + b
            # Apply tanh activation for all layers except the last
            if i < len(js_model["layers"]) - 1:
                x = np.tanh(x)

        action_manual = int(np.argmax(x))

        if int(action_sb3) != action_manual:
            errors.append((obs[:3], int(action_sb3), action_manual))

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/sumo_final",
                        help="Path to trained model (without .zip)")
    parser.add_argument("--output", type=str, default="models/policy.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    # Resolve paths relative to CWD
    model_path = os.path.abspath(args.model)
    output_path = os.path.abspath(args.output)

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    print("Extracting policy weights...")
    layers = extract_policy_weights(model)

    print("Building JS-compatible model...")
    js_model = build_js_model(layers)

    # Print architecture info
    for i, layer in enumerate(js_model["layers"]):
        w_shape = np.array(layer["weight"]).shape
        act = "tanh" if i < len(js_model["layers"]) - 1 else "none (logits)"
        print(f"  Layer {i}: {w_shape[1]} → {w_shape[0]}  activation: {act}")

    # Verify
    print("\nVerifying export correctness (10 random inputs)...")
    errors = verify_export(model, js_model)
    if errors:
        print(f"  WARNING: {len(errors)}/10 mismatches!")
        for obs, sb3_act, manual_act in errors[:3]:
            print(f"    obs[:3]={obs}, sb3={sb3_act}, manual={manual_act}")
    else:
        print("  ✓ All 10 test inputs match perfectly.")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(js_model, f)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nExported to {output_path} ({size_kb:.1f} KB)")
    print("Ready to load in the browser arena!")


if __name__ == "__main__":
    main()
