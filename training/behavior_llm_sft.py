import os
import sys
import os
import torch
import time

# Resolve project_root = one level up from this script
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Prepend it so your imports see the entire repo
sys.path.insert(0, project_root)

import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorWithPadding
from transformers import EarlyStoppingCallback, BitsAndBytesConfig, TrainingArguments
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
from dataclasses import dataclass
from models.behavior_gemma7b import GemmaModular
from accelerate import Accelerator
from trl import SFTTrainer, SFTConfig
from utils.logging import init_wandb
from datasets import load_dataset, concatenate_datasets, load_from_disk
import glob, shutil
from bitsandbytes.optim import Adam8bit
from dotenv import load_dotenv
import wandb
import inspect, textwrap

# -----------------------------

BACKBONE_ID = "google/gemma-7b"
hf_token = os.getenv("HF_API_KEY", None)
load_in_8bit = True
include_test = False
learning_rate = 2e-4
context_length = 1024

accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(
    project_name="gemma_behavior_sft",
    config={},
    init_kwargs={"wandb": {"entity": "thelemonpig-cyberkind"}}
)

rank = accelerator.local_process_index
world_size = accelerator.num_processes

# A tiny stagger to keep the prints readable
time.sleep(rank * 0.1)

print(f"[Rank {rank}/{world_size}] 🚀 after Accelerator()", flush=True)

# Now print which CUDA device this rank thinks it has
print(f"[Rank {rank}/{world_size}] using device {accelerator.device}"
      f" (torch.cuda.current_device() -> {torch.cuda.current_device()})", flush=True)

print(f"[Rank {rank}/{world_size}] ✅ ready to load model", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    BACKBONE_ID,
    token=hf_token
    )
EOS_TOKEN = tokenizer.eos_token

if load_in_8bit:
    quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # bnb_8bit_compute_dtype=torch.bfloat16,
    bnb_8bit_compute_dtype=torch.float32,   # <- critical change
    llm_int8_threshold=6.0,            # optional tuning threshold
    # llm_int8_has_fp16_weight=False,    # optional
    )
    rank = accelerator.local_process_index
    print(f"[Rank {rank}] about to load backbone on GPU {torch.cuda.current_device()}")
    print(f"[Rank {rank}/{world_size}] 📦 loading model now…", flush=True)
    backbone = AutoModelForCausalLM.from_pretrained(
    BACKBONE_ID,
    quantization_config=quant_config,
    # device_map=accelerator.device, #'auto',   # Uncomment this unless you are using DDP
    output_hidden_states=True,
    token=hf_token,
    )
    print(f"[Rank {rank}/{world_size}] 📦 done loading model", flush=True)
    print(f"[Rank {rank}] finished loading backbone")
else:
    backbone = AutoModelForCausalLM.from_pretrained(
        BACKBONE_ID,
        torch_dtype=torch.bfloat16,
        # device_map=accelerator.device,  # Uncomment this unless you are using DDP
        output_hidden_states=True,
        token=hf_token,
    )
m16 = AutoModelForCausalLM.from_pretrained("google/gemma-7b",
                                           torch_dtype="bfloat16").cuda()
# --- quick BF16 sanity‑check (no quant) -----------------
tok = tokenizer("hello", return_tensors="pt").to(m16.device)  # tokenise & move to GPU
with torch.no_grad():
    out = m16(**tok)
print("→ bf16 no-quant forward:", out.logits[0, 0, :5])
with torch.no_grad():
    out = backbone(**tok)
print("→ with quant forward:", out.logits[0, 0, :5])
# --------------------------------------------------------
model = GemmaModular(backbone)
model.config = AutoConfig.from_pretrained(
    BACKBONE_ID,
    quantization_config=quant_config,
    # device_map=accelerator.device, #'auto',   # Uncomment this unless you are using DDP
    output_hidden_states=True,
    token=hf_token,
    )
for layer in model.backbone_layers:
    attn = layer.self_attn
    attn.q_proj.register_forward_hook(lambda m, inp, out: out.to(torch.bfloat16))
    attn.k_proj.register_forward_hook(lambda m, inp, out: out.to(torch.bfloat16))
    attn.v_proj.register_forward_hook(lambda m, inp, out: out.to(torch.bfloat16))
print(f"[Rank {rank}] Gemma modular made on {accelerator.device}")
model.to(accelerator.device)

# model = accelerator.prepare(model)
# Use 8-bit Adam for trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"[Rank {rank}] parameters catalogued on {accelerator.device}")
optimizer = Adam8bit(trainable_params, lr=learning_rate)
print(f"[Rank {rank}] optimizer built on {accelerator.device}")
# optimizer = accelerator.prepare(optimizer)
# Use this if you are initializing your own optimizer – due to funky 8bit stuff I am leaving it to SFTTrainer

print(f"[Rank {rank}] model & optimizer ready on {accelerator.device}")

# Uncomment these lines to check loading for DDP
local_rank = accelerator.local_process_index  # 0, 1, 2, or 3
device = accelerator.device                   # torch.device("cuda:0") for rank0, "cuda:1" for rank1, etc.
print(f"Rank {local_rank} → Device {device}")
print(torch.cuda.memory_summary(device))


# ------------- sanity pass ---------------------------------

dummy = tokenizer("quick test", return_tensors="pt")
dummy["input_ids"]      = dummy["input_ids"].to(model.lm_head.weight.device)
dummy["attention_mask"] = dummy["attention_mask"].to(dtype=torch.bool,   # ← cast
                                                     device=model.lm_head.weight.device)
model.eval()
with torch.no_grad():
    out = model(**dummy)
print("[sanity] logits shape:", out["logits"].shape)   # expect (1, T, vocab)
model.train()
# -----------------------------------------------------------

# -----------------------------

# 3.  Load datasets
# -----------------------------
alpaca       = load_dataset("yahma/alpaca-cleaned", split="train")       # 52 K
dolly        = load_dataset("databricks/databricks-dolly-15k", split="train") # 15 K
openassistant= load_dataset("OpenAssistant/oasst2", split="train") #128 K

# -----------------------------
# 4.  Configure and launch training
# -----------------------------
# Ensure the dataset yields input_ids/labels for SFTTrainer
def preprocess(example):
    # print("✂️  In preprocess, example keys:", list(example.keys()))
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
        text=text,                       # name it explicitly
        truncation=True,
        max_length=context_length,
        padding="max_length",
    )
    # For causal LM, labels = input_ids (standard SFT)    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# only main process does the heavy map/filter
# if accelerator.is_local_main_process:
combined = concatenate_datasets([alpaca, dolly, openassistant]).shuffle(seed=42)
splits   = combined.train_test_split(test_size=0.1, seed=42)
train_raw, eval_raw = splits["train"], splits["test"]

# filter
train_raw = train_raw.filter(lambda ex: bool(ex.get("text")))
eval_raw  = eval_raw.filter(lambda ex: bool(ex.get("text")))

# tokenize
processed_train = train_raw.map(preprocess, batched=False)
processed_eval  = eval_raw.map(preprocess, batched=False)

# # save to disk for the others
# processed_train.save_to_disk("train_ds")
# processed_eval.save_to_disk("eval_ds")
# # accelerator.wait_for_everyone()

# processed_train = load_from_disk("train_ds")
# processed_eval  = load_from_disk("eval_ds")

# processed_train = load_dataset("train_ds")
# processed_eval  = load_dataset("eval_ds")

# Dummy batch
if include_test:
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

# Define SFT training configuration
# sft_config = SFTConfig(
#     per_device_train_batch_size=6,               # adjust as needed for your hardware
#     gradient_accumulation_steps=1,     # adjust to simulate larger batches if needed
#     learning_rate=learning_rate,                # match your optimizer setting
#     num_train_epochs=3,                # set number of epochs
#     logging_steps=50,                  # log every N steps
#     save_strategy="steps",             # checkpoint by steps
#     save_steps=100,                    # every 100 steps
#     save_total_limit=1,                # keep only the most recent
#     report_to="wandb",                 # enable reporting to Weights & Biases
#     gradient_checkpointing=True,
#     lr_scheduler_type="cosine",    # or "linear", "polynomial", etc.
#     warmup_ratio=0.1,              # warm up for 10% of total training steps
#     max_grad_norm=1.0,            # gradient clipping
#     eval_accumulation_steps=2,  # accumulate eval results for better metrics
#     output_dir=os.path.join(project_root, "sft_output"),  # where to save checkpoints
#     optim="adamw_bnb_8bit",
# )

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    logging_steps=50,
    num_train_epochs=3,
    gradient_accumulation_steps=1,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    output_dir=os.path.join(project_root, "sft_output"),  # where to save checkpoints
)



total = sum(p.numel() for p in trainable_params)
print(f"Optimizing {total/1e6:.1f}M params")
print("Using batch size:", training_args.per_device_train_batch_size)


base_collator = DataCollatorWithPadding(tokenizer)

def collate(features):
    batch = base_collator(features)
    batch["attention_mask"] = (batch["input_ids"] != tokenizer.pad_token_id)
    return batch

# Initialize the SFT trainer with train and eval datasets
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_eval,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    data_collator=collate,
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
        print("Using batch size:", training_args.per_device_train_batch_size)
        last_ckpts = sorted(glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")), key=os.path.getmtime)
        print("Using batch size:", training_args.per_device_train_batch_size)
        resume = last_ckpts[-1] if last_ckpts else None
        trainer.train(resume_from_checkpoint=resume)
        break
    except Exception as e:
        print(f"Interrupted ({e}), restarting from last checkpoint...")
        cleanup_checkpoints(training_args.output_dir, keep=1)
        attempt += 1
        if attempt >= 5:
            raise


print("Training complete, shutting down pod.")
print(f"[Rank {rank}] Training complete on {accelerator.device}")
time.sleep(120) # wait_for_everyone doesn't work so let's hope that 2 minutes is enough for everyone to finish
sys.exit(0)
# Final barrier and pod shutdown
# accelerator.wait_for_everyone()
# if accelerator.is_main_process:
#     print("Training complete, shutting down pod.")
#     sys.exit(0)