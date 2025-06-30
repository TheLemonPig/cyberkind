import torch
import torch.nn as nn
from transformers import AutoTokenizer
from dataclasses import dataclass
from models.behavior_llm import build_model
from accelerate import Accelerator
from trl import SFTTrainer, SFTConfig
from utils.logging import init_wandb

# -----------------------------

BACKBONE_ID = "google/gemma-7b"

accelerator = Accelerator()

if accelerator.is_main_process:
    init_wandb()
accelerator.wait_for_everyone()

tokenizer = AutoTokenizer.from_pretrained(BACKBONE_ID)
EOS_TOKEN = tokenizer.eos_token
model = build_model()

# -----------------------------

# Dummy batch
text = "Alice thinks Bob believes Carol loves Dave. Why?"
batch = tokenizer(text, return_tensors="pt").to(model.lm_head.weight.device)
out = model(**batch)
print(out["logits"].shape)  # (1, seq_len, vocab)

# Optimizer only sees module + lm_head params
trainables = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable params: {sum(p.numel() for p in trainables)/1e6:.1f}M")
optim = torch.optim.AdamW(trainables, lr=2e-4)

# One training step example
targets = batch["input_ids"]
loss_fn = nn.CrossEntropyLoss()

logits = model(**batch)["logits"]
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = targets[:, 1:].contiguous()
loss = loss_fn(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
loss.backward()
optim.step()
optim.zero_grad()
print("step loss:", loss.item())