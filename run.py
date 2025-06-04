from envs.gridworld.env import GridWorldEnv
from models.transformer import CyberKindModel
from training.pretrain import run_pretraining
from utils.logging import init_wandb

import torch

def evaluate_behavior(env, model, episodes=20, max_steps=50):
    successes = 0
    total_steps = []

    for _ in range(episodes):
        state = env.reset()
        for step in range(max_steps):
            state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
            with torch.no_grad():
                output = model(state_tensor)
                action = torch.argmax(output["behavior"]).item()  # greedy action

            state, reward, done, _ = env.step(action)
            if done:
                successes += 1
                total_steps.append(step + 1)
                break
        else:
            total_steps.append(max_steps)

    avg_steps = sum(total_steps) / len(total_steps)
    print(f"[Behavior Evaluation] Success rate: {successes}/{episodes} | Avg steps: {avg_steps:.2f}")

def evaluate_prediction(env, model, steps=100):
    correct = 0

    state = env.reset()
    for _ in range(steps):
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
        with torch.no_grad():
            output = model(state_tensor)
            pred_index = torch.argmax(output["prediction"]).item()

        action = torch.argmax(output["behavior"]).item()
        next_state, _, done, _ = env.step(action)
        true_index = torch.argmax(
            torch.tensor(next_state.flatten(), dtype=torch.float32)
        ).item()

        if pred_index == true_index:
            correct += 1

        if done:
            state = env.reset()
        else:
            state = next_state

    print(f"[Prediction Evaluation] Accuracy: {correct}/{steps} ({100 * correct / steps:.1f}%)")

if __name__ == "__main__":
    # init_wandb(project_name="cyberkind-gridworld")

    size=3
    env = GridWorldEnv(size=size)
    model = CyberKindModel(input_dim=size*size, num_actions=4, embed_dim=64, num_heads=2, num_layers=2)

    run_pretraining(env, model, steps=1000, max_episode_length=50)

    evaluate_behavior(env, model, episodes=20, max_steps=100)
    evaluate_prediction(env, model, steps=100)
    # wandb.finish()
