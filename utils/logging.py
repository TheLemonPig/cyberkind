import wandb

def init_wandb(project_name="cyberkind"):
    wandb.init(project=project_name)