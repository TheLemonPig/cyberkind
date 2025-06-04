import os
from dotenv import load_dotenv
import wandb

def init_wandb(project_name="cyberkind"):
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project=project_name)
