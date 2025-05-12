import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import time

from envs.gridworld.env import GridWorldEnv
from models.transformer_base import CyberKindModel

def visualize_untrained_model(steps=5, sleep_time=0.5):
    env = GridWorldEnv()
    model = CyberKindModel()
    obs = env.reset()

    print("\n=== Untrained Model Visualization ===\n")
    for step in range(steps):
        obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32)
        
        with torch.no_grad():
            output = model(obs_tensor)
            behavior_logits = output["behavior"]
            prediction_logits = output["prediction"]

        # Visualize the agent's position on the grid
        grid = np.array(obs)
        print(f"Step {step}")
        print("Grid:")
        print(grid)
        print("Behavior logits:", behavior_logits.numpy())
        print("Prediction logits:", prediction_logits.numpy())
        print("-" * 40)

        # Pick a random action (untrained)
        action = np.random.choice(4)
        obs, reward, done, _ = env.step(action)

        time.sleep(sleep_time)
        if done:
            print("Reached goal!")
            break

if __name__ == "__main__":
    visualize_untrained_model()