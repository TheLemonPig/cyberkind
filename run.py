from envs.gridworld.env import GridWorldEnv
from models.transformer_base import CyberKindTransformer
from training.pretrain import run_pretraining
from utils.logging import init_wandb

if __name__ == "__main__":
    init_wandb(project_name="cyberkind-gridworld")

    env = GridWorldEnv()
    model = CyberKindTransformer()

    run_pretraining(env, model)