from envs.gridworld.env import GridWorldEnv
from models.transformer_base import CyberKindModel
from training.pretrain import run_pretraining
from utils.logging import init_wandb

import torch

if __name__ == "__main__":
    init_wandb(project_name="cyberkind-gridworld")

    env = GridWorldEnv()
    model = CyberKindModel()

    run_pretraining(env, model, steps=1000)