import torch

from envs.gridworld.env import GridWorldEnv
from models.transformer import CyberKindModel


def test_model_forward():
    env = GridWorldEnv(size=2)
    model = CyberKindModel(input_dim=4, num_actions=4, embed_dim=16, num_heads=1, num_layers=1)
    state = torch.tensor(env.reset().flatten(), dtype=torch.float32)
    output = model(state)
    assert "behavior" in output and "prediction" in output
    assert output["behavior"].shape[0] == 4
    assert output["prediction"].shape[0] == 4
