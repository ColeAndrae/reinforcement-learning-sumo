#!/usr/bin/env python3
"""export_to_js.py v4"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO

def main():
    model = PPO.load(os.path.join(os.getcwd(), 'models', 'sumo_final'))
    layers = []
    for name, param in model.policy.mlp_extractor.policy_net.named_parameters():
        layers.append(('policy.'+name, param.detach().cpu().numpy()))
    for name, param in model.policy.action_net.named_parameters():
        layers.append(('action.'+name, param.detach().cpu().numpy()))

    js = {'layers': []}
    i = 0
    while i < len(layers):
        if 'weight' in layers[i][0] and i+1 < len(layers) and 'bias' in layers[i+1][0]:
            js['layers'].append({'weight': layers[i][1].tolist(), 'bias': layers[i+1][1].tolist()})
            i += 2
        else:
            i += 1

    # Verify
    mismatches = 0
    for _ in range(200):
        obs = np.random.uniform(-1.5, 1.5, 10).astype(np.float32)
        sb3_a, _ = model.predict(obs, deterministic=True)
        x = obs.astype(np.float64)
        for j, l in enumerate(js['layers']):
            x = x @ np.array(l['weight']).T + np.array(l['bias'])
            if j < len(js['layers'])-1: x = np.tanh(x)
        if int(sb3_a) != int(np.argmax(x)): mismatches += 1

    print(f'Architecture: {" → ".join(str(np.array(l["weight"]).shape[1]) for l in js["layers"])} → {np.array(js["layers"][-1]["weight"]).shape[0]}')
    print(f'Verification: {200-mismatches}/200 match' + (' ✓' if mismatches==0 else f' ✗ ({mismatches} mismatches!)'))

    out = os.path.join(os.getcwd(), 'models', 'policy.json')
    with open(out, 'w') as f:
        json.dump(js, f, separators=(',',':'))
    print(f'Exported: {out} ({os.path.getsize(out)/1024:.0f} KB)')

if __name__ == '__main__':
    main()
