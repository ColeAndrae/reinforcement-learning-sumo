#!/usr/bin/env python3
"""
export_to_js.py v3 — Export trained policy to JSON and verify correctness.

Extracts weight matrices from the PPO MlpPolicy and verifies the
manual forward pass matches SB3's predict() on 100 random inputs.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO


def extract_policy_weights(model):
    policy = model.policy
    layers = []
    for name, param in policy.mlp_extractor.policy_net.named_parameters():
        layers.append({
            "name": f"policy_net.{name}",
            "shape": list(param.shape),
            "data": param.detach().cpu().numpy().tolist(),
        })
    for name, param in policy.action_net.named_parameters():
        layers.append({
            "name": f"action_net.{name}",
            "shape": list(param.shape),
            "data": param.detach().cpu().numpy().tolist(),
        })
    return layers


def build_js_model(layers):
    processed = []
    i = 0
    while i < len(layers):
        w = layers[i]
        b = layers[i + 1] if i + 1 < len(layers) else None
        if "weight" in w["name"] and b and "bias" in b["name"]:
            processed.append({"weight": w["data"], "bias": b["data"]})
            i += 2
        else:
            i += 1
    return {"architecture": "mlp", "activation": "tanh", "layers": processed}


def manual_forward(js_model, obs):
    """Replicate the JS forward pass in Python."""
    x = obs.copy().astype(np.float64)
    for i, layer in enumerate(js_model["layers"]):
        w = np.array(layer["weight"])
        b = np.array(layer["bias"])
        x = x @ w.T + b
        if i < len(js_model["layers"]) - 1:
            x = np.tanh(x)
    return int(np.argmax(x))


def verify_export(model, js_model, n_tests=100):
    """Verify manual forward pass matches SB3 on random inputs."""
    mismatches = 0
    for _ in range(n_tests):
        obs = np.random.uniform(-1.5, 1.5, size=(14,)).astype(np.float32)
        sb3_action, _ = model.predict(obs, deterministic=True)
        manual_action = manual_forward(js_model, obs)
        if int(sb3_action) != manual_action:
            mismatches += 1
    return mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/sumo_final")
    parser.add_argument("--output", type=str, default="models/policy.json")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    output_path = os.path.abspath(args.output)

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    print("Extracting weights...")
    layers = extract_policy_weights(model)
    js_model = build_js_model(layers)

    for i, layer in enumerate(js_model["layers"]):
        w = np.array(layer["weight"])
        act = "tanh" if i < len(js_model["layers"]) - 1 else "none (logits)"
        print(f"  Layer {i}: {w.shape[1]} → {w.shape[0]}  [{act}]")

    print(f"\nVerifying on 100 random inputs...")
    mismatches = verify_export(model, js_model, n_tests=100)
    if mismatches:
        print(f"  ✗ {mismatches}/100 MISMATCHES — export may be broken!")
    else:
        print(f"  ✓ 100/100 match perfectly")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(js_model, f, separators=(',', ':'))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nExported: {output_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
