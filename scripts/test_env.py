#!/usr/bin/env python3
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.sumo_env import SumoEnv, heuristic_aggressive, heuristic_flanker, heuristic_positional

def main():
    env = SumoEnv()
    obs, _ = env.reset()
    print(f'Obs: {obs.shape}, Actions: {env.action_space.n}')
    assert obs.shape == (10,) and env.action_space.n == 9

    for name, fn in [('Aggressive', heuristic_aggressive), ('Flanker', heuristic_flanker), ('Positional', heuristic_positional)]:
        e = SumoEnv(opponent_policy=fn)
        w = 0
        for _ in range(50):
            o, _ = e.reset(); done = False
            while not done:
                o, _, t, tr, info = e.step(e.action_space.sample())
                done = t or tr
            if info.get('winner') == 0: w += 1
        print(f'  Random vs {name}: {w}/50')

    # Perf
    start = time.time(); steps = 0
    for _ in range(200):
        o, _ = env.reset(); done = False
        while not done:
            o, _, t, tr, _ = env.step(env.action_space.sample())
            done = t or tr; steps += 1
    fps = steps / (time.time() - start)
    print(f'  {fps:,.0f} steps/sec → 2M ≈ {2e6/fps/60:.1f} min')
    print('All OK.')

if __name__ == '__main__':
    main()
