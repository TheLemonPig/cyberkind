import torch
import torch.nn.functional as F
from tqdm import tqdm

def run_pretraining(env, model, steps=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in tqdm(range(steps)):
        obs = env.reset()
        obs = torch.tensor(obs.flatten(), dtype=torch.float32)
        outputs = model(obs)

        loss = F.mse_loss(outputs["behavior"], obs)  # Dummy self-supervised task

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()