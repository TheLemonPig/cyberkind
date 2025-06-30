import os, sys

# Resolve project_root = one level up from this script
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Prepend it so your imports see the entire repo
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from models.behavior_llm import GemmaWithModule
from accelerate import Accelerator
from trl import SFTTrainer, SFTConfig
from utils.logging import init_wandb
from datasets import load_dataset, concatenate_datasets

# -----------------------------

BACKBONE_ID = "google/gemma-7b"
hf_token = os.getenv("HF_API_KEY", None)

accelerator = Accelerator()

if accelerator.is_main_process:
    init_wandb()
accelerator.wait_for_everyone()

tokenizer = AutoTokenizer.from_pretrained(
    BACKBONE_ID,
    use_auth_token=hf_token
    )
EOS_TOKEN = tokenizer.eos_token

    
backbone = AutoModelForCausalLM.from_pretrained(
    BACKBONE_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_8bit=True,
    output_hidden_states=True,
    use_auth_token=hf_token,
)
model = GemmaWithModule(backbone, module_depth=8, module_dim=1024)
model.to(accelerator.device)

for p in model.parameters():
    p.requires_grad = False


# -----------------------------

# 3.  Load datasets
# -----------------------------
alpaca       = load_dataset("tatsu-lab/alpaca-cleaned")       # 52 K
dolly        = load_dataset("databricks/databricks-dolly-15k")# 15 K
openassistant= load_dataset("OpenAssistant/oasst2", split="all")#128 K

# Optional: sample down to token budget if needed
# e.g., retain first 200 K examples
combined = concatenate_datasets([alpaca, dolly, openassistant]).shuffle(seed=42).select(range(200_000))


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