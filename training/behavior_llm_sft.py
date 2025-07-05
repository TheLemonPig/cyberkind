import os
import sys

# maximum context length (tokens) to prevent overflow during tokenization
context_length = 1024

# Resolve project_root = one level up from this script
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Prepend it so your imports see the entire repo
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from transformers import EarlyStoppingCallback
from dataclasses import dataclass
from models.behavior_llm_old import GemmaWithModule
from accelerate import Accelerator
from trl import SFTTrainer, SFTConfig
from utils.logging import init_wandb
from datasets import load_dataset, concatenate_datasets
import glob, shutil
from bitsandbytes.optim import Adam8bit

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
combined = concatenate_datasets([alpaca, dolly, openassistant]).shuffle(seed=42) #.select(range(200_000))

# -----------------------------
# 3b. Split into train/validation
# -----------------------------
splits = combined.train_test_split(test_size=0.1, seed=42)
train_raw = splits["train"]
eval_raw  = splits["test"]

# Remove examples whose raw token count exceeds context_length
train_raw = train_raw.filter(
    lambda ex: len(tokenizer(ex.get("text", "") or "", add_special_tokens=False).input_ids) <= context_length
)
eval_raw = eval_raw.filter(
    lambda ex: len(tokenizer(ex.get("text", "") or "", add_special_tokens=False).input_ids) <= context_length
)


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

# -----------------------------
# 4.  Configure and launch training
# -----------------------------
# Ensure the dataset yields input_ids/labels for SFTTrainer
def preprocess(example):
    # Try to get prompt + completion for each dataset
    # Fallback to text if necessary
    text = None
    if "text" in example:
        text = example["text"]
    elif "prompt" in example and "completion" in example:
        text = example["prompt"] + example["completion"]
    elif "instruction" in example and "output" in example:
        text = example["instruction"] + example["output"]
    else:
        # fallback to string of all values
        text = " ".join(str(v) for v in example.values() if isinstance(v, str))
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=context_length,
        padding="max_length",
    )
    # For causal LM, labels = input_ids (standard SFT)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Map preprocessing separately for train and eval
processed_train = train_raw.map(preprocess, batched=False)
processed_eval  = eval_raw.map(preprocess,  batched=False)

# Enable gradient checkpointing to reduce activation memory
model.gradient_checkpointing_enable()

# Define SFT training configuration
sft_config = SFTConfig(
    train_batch_size=8,               # adjust as needed for your hardware
    gradient_accumulation_steps=1,     # adjust to simulate larger batches if needed
    learning_rate=2e-4,                # match your optimizer setting
    num_train_epochs=3,                # set number of epochs
    logging_steps=50,                  # log every N steps
    evaluation_strategy="steps",       # run evaluation every eval_steps
    eval_steps=500,                    # adjust to your preferred frequency
    save_strategy="steps",             # checkpoint by steps
    save_steps=100,                    # every 100 steps
    save_total_limit=1,                # keep only the most recent
    report_to="wandb",                 # enable reporting to Weights & Biases
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",    # or "linear", "polynomial", etc.
    warmup_ratio=0.1,              # warm up for 10% of total training steps
    max_grad_norm=1.0,            # gradient clipping
    eval_accumulation_steps=2,  # accumulate eval results for better metrics
    output_dir=os.path.join(project_root, "sft_output")  # where to save checkpoints
)

# Use 8-bit Adam for trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = Adam8bit(trainable_params, lr=sft_config.learning_rate)

# Initialize the SFT trainer with train and eval datasets
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=processed_train,
    eval_dataset=processed_eval,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    data_collator=default_data_collator,
    optimizers=(optimizer, None),  # use 8-bit Adam
)

# -----------------------------
# 5.  Automatic-restart training loop for spot instances
# -----------------------------
def cleanup_checkpoints(output_dir, keep=1):
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")), key=os.path.getmtime)
    for ckpt in ckpts[:-keep]:
        shutil.rmtree(ckpt)

attempt = 0
while True:
    try:
        # find last checkpoint if any
        last_ckpts = sorted(glob.glob(os.path.join(sft_config.output_dir, "checkpoint-*")), key=os.path.getmtime)
        resume = last_ckpts[-1] if last_ckpts else None
        trainer.train(resume_from_checkpoint=resume)
        break
    except Exception as e:
        print(f"Interrupted ({e}), restarting from last checkpoint...")
        cleanup_checkpoints(sft_config.output_dir, keep=1)
        attempt += 1
        if attempt >= 5:
            raise

# Final barrier and pod shutdown
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    print("Training complete, shutting down pod.")
    sys.exit(0)