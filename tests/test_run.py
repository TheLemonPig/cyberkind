import pytest

from envs.gridworld.env import GridWorldEnv
from models.transformer import CyberKindModel
from training.pretrain import run_pretraining
from run import evaluate_behavior, evaluate_prediction


def test_run_minimal():
    size = 2
    env = GridWorldEnv(size=size)
    model = CyberKindModel(
        input_dim=size * size,
        num_actions=4,
        embed_dim=16,
        num_heads=1,
        num_layers=1,
    )

    # run a very small training loop to ensure the code executes
    run_pretraining(env, model, steps=2, max_episode_length=5)

    # ensure evaluation functions run without error
    evaluate_behavior(env, model, episodes=1, max_steps=5)
    evaluate_prediction(env, model, steps=5)
