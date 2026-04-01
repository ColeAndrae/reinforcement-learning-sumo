# Cube Sumo — RL Self-Play Training Pipeline

Two cubes battle in a circular ring, trying to push each other out. Agents learn
to fight through PPO self-play, then deploy to a polished Three.js browser arena.

## Project Structure

```
sumo-rl/
├── envs/
│   ├── __init__.py
│   ├── sumo_env.py          # Core physics + Gymnasium env
│   └── self_play_env.py     # Self-play wrapper with opponent pool
├── scripts/
│   ├── test_env.py           # Smoke tests + performance benchmark
│   ├── train.py              # PPO self-play training loop
│   └── export_to_js.py       # Export weights → JSON for browser
├── models/                   # Saved checkpoints + final model
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install gymnasium stable-baselines3 numpy
```

### 2. Test the environment
```bash
cd scripts
python test_env.py
```

### 3. Train (M2 MacBook: ~15-30 min for 500K steps)
```bash
python train.py --timesteps 500000
```

### 4. Export to browser
```bash
python export_to_js.py --model ../models/sumo_final
```
This produces `models/policy.json` (~50-100 KB) which loads into the
Three.js arena for autonomous cube battles.

## Architecture

### Environment (SumoEnv)
- **Observation** (12 dims): normalized positions, velocities, relative
  opponent state, distances to ring edge, angles
- **Action** (5 discrete): no-op, ±X, ±Z force
- **Physics**: matched exactly to the browser arena (ring radius 5.5,
  friction 0.92, bounce 0.3, cube-to-cube AABB collision + impulse)
- **Reward**: +10 win / -10 loss, with shaping for edge pressure +
  engagement distance + time penalty

### Self-Play (SelfPlaySumoEnv)
- Maintains a pool of up to 10 past policy checkpoints
- Swaps opponent every 50 episodes from the pool
- 20% chance of random opponent (prevents overfitting)
- Random side-swapping for left/right symmetry

### Training (PPO)
- Policy: 2×128 hidden layers with tanh activation
- Standard PPO hyperparameters (lr=3e-4, clip=0.2, entropy=0.01)
- Checkpoints every 25K steps → fed into opponent pool

### Browser Inference
The exported `policy.json` contains raw weight matrices.
A 15-line JS forward pass (matmul + tanh) runs the agent at 60fps
with zero dependencies.
