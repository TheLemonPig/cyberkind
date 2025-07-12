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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, default_data_collator
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
print(textwrap.dedent(inspect.getsource(GemmaRotaryEmbedding.forward)))

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

# if accelerator.is_main_process:
#     load_dotenv()
#     print(f"[Rank 0/4] calling init_wandb()")
#     wandb.login(key=os.getenv("WANDB_API_KEY"))
#     wandb.init(project='cyberkind', resume='allow')
# else:
#     # start a no-op run – returns immediately, avoids network + disk traffic
#     print(f"[Rank {accelerator.local_process_index}/4] skipping init_wandb()")
#     # wandb.init(mode="disabled")

# print(f"[Rank {accelerator.local_process_index}/4] just before wait_for_everyone()", flush=True)
# accelerator.wait_for_everyone()
# print(f"[Rank {accelerator.local_process_index}/4] just before wait_for_everyone()", flush=True)


# print(f"[Rank {rank}/{world_size}] ⏳ passed wait_for_everyone()", flush=True)

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
    bnb_8bit_compute_dtype=torch.bfloat16,
    llm_int8_threshold=6.0,            # optional tuning threshold
    llm_int8_has_fp16_weight=False,    # optional
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
    from transformers.models.gemma.modeling_gemma import GemmaAttention
    print(textwrap.dedent(
    inspect.getsource(GemmaAttention.forward)
    ).splitlines()[0:20])
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
def check_rotary(model):
    missing = []
    for idx, bl in enumerate(model.backbone_layers):
        attn = bl.self_attn
        dev  = next(attn.parameters()).device
        if getattr(attn, "rotary_emb", None) is None:
            missing.append((idx, str(dev)))
    if torch.distributed.get_rank() == 0:   # only rank-0 prints
        if missing:
            print(">>> Layers missing rotary_emb:")
            for idx, dev in missing:
                print(f"    layer {idx} on {dev}")
        else:
            print(">>> All backbone layers have rotary_emb")

def inspect_rotary(model):
    print("\n[rotary diagnostic]")
    bad = []
    for idx, bl in enumerate(model.backbone_layers):
        attn = bl.self_attn
        # ① does the attribute exist?
        has_attr = hasattr(attn, "rotary_emb")
        obj      = getattr(attn, "rotary_emb", None)
        # ② correct type?  Gemma’s class name is "GemmaRotaryEmbedding"
        cls_name = type(obj).__name__ if has_attr else "-"
        # ③ is it callable and returns (cos,sin)?
        ok_call  = False
        if callable(obj):
            try:
                out = obj(2, torch.float32, torch.device("cpu"))
                ok_call = isinstance(out, tuple) and len(out) == 2
            except Exception:
                pass
        if not (has_attr and ok_call):
            bad.append((idx, cls_name))
    if bad:
        print("missing or broken rotary embeddings in layers:")
        for idx, cls_name in bad:
            print(f"  layer {idx:<2d}  rotary_emb={cls_name}")
    else:
        print("✓ every layer has a working rotary_emb\n")
def attach_bnb_keep(model):
    """
    Register a 1-byte buffer on every GemmaRotaryEmbedding so BnB’s
    replacement pass keeps the module, even after deepcopy().
    """
    for mod in model.modules():
        if isinstance(mod, GemmaRotaryEmbedding):
            if not hasattr(mod, "bnb_keep"):
                mod.register_buffer("bnb_keep", torch.zeros(1), persistent=False)
def count_missing_rotary(msg, model):
    miss = sum(
        1 for bl in model.model.layers  # 28 Gemma blocks
        if getattr(bl.self_attn, "rotary_emb", None) is None
    )
    print(f"{msg:<15}  missing={miss}")
def debug_rotary(model, n_tokens: int = 4):
    """
    Checks every Gemma block for a working rotary embedding.
    • OK   – attribute exists and call returns (cos, sin)
    • MISS – attribute missing
    • ERR  – call raises / returns something unexpected
    """
    # 1) locate the list of blocks
    if hasattr(model, "backbone_layers"):        # GemmaModular
        layers = model.backbone_layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):  # raw Gemma
        layers = model.model.layers
    else:
        raise ValueError("Cannot find Gemma layers on model")

    print("\n[rotary debug]")
    for idx, block in enumerate(layers):
        attn = block.self_attn
        tag, note = "OK", ""
        if not hasattr(attn, "rotary_emb"):
            tag, note = "MISS", "attribute missing"
        else:
            try:
                device = next(attn.parameters()).device
                hidden = torch.zeros(
                    1, n_tokens, 1, attn.head_dim,
                    device=device,
                    dtype=attn.q_proj.weight.dtype,
                )
                cos, sin = attn.rotary_emb(hidden, None)
                if not (isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor)):
                    tag, note = "ERR", "call did not return tensors"
            except Exception as e:
                tag, note = "ERR", repr(e)

        print(f"layer {idx:02d}: {tag:4s} {note}")
    print()  # newline
debug_rotary(backbone)        # immediately after from_pretrained

# (A) Plain load
fp_backbone = AutoModelForCausalLM.from_pretrained(BACKBONE_ID, torch_dtype=torch.bfloat16, token=hf_token)
count_missing_rotary("fp16 load", fp_backbone)   # expect 0
debug_rotary(fp_backbone)        # immediately after from_pretrained

del fp_backbone    # free VRAM
torch.cuda.empty_cache()

# (B) 8-bit load
bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
bnb_backbone = AutoModelForCausalLM.from_pretrained(BACKBONE_ID, quantization_config=bnb_config, token=hf_token)
count_missing_rotary("8-bit load", bnb_backbone)
debug_rotary(bnb_backbone)        # immediately after from_pretrained

backbone.gradient_checkpointing_enable()
print(f"[Rank {rank}] gradient checkpoint ready on {accelerator.device}")
attach_bnb_keep(backbone)      # ← add this line
# This is needed to keep GemmaRotaryEmbedding modules in the model
def check_bnb_keep(model):
    """
    Check if the bnb_keep buffer is present in GemmaRotaryEmbedding modules.
    """
    missing = []
    for idx, mod in enumerate(model.modules()):
        if isinstance(mod, GemmaRotaryEmbedding):
            if not hasattr(mod, "bnb_keep"):
                missing.append(idx)
    if torch.distributed.get_rank() == 0:   # only rank-0 prints
        if missing:
            print(">>> Layers missing bnb_keep:")
            for idx in missing:
                print(f"    layer {idx}")
        else:
            print(">>> All GemmaRotaryEmbedding layers have bnb_keep")
check_bnb_keep(backbone)
# Check if the GemmaRotaryEmbedding modules have bnb_keep
debug_rotary(backbone)        # immediately after from_pretrained

model = GemmaModular(backbone)
debug_rotary(model)        # immediately after from_pretrained

check_rotary(model)
inspect_rotary(model)
attn0 = model.backbone_layers[0].self_attn
# seq_len  = 4
# pos_ids  = torch.arange(seq_len, device=attn0.q_proj.weight.device)
# print("rotary call returns:", attn0.rotary_emb(seq_len, pos_ids))
model.config = AutoConfig.from_pretrained(
    BACKBONE_ID,
    quantization_config=quant_config,
    # device_map=accelerator.device, #'auto',   # Uncomment this unless you are using DDP
    output_hidden_states=True,
    token=hf_token,
    )
check_rotary(model)
print(f"[Rank {rank}] Gemma modular made on {accelerator.device}")
model.to(accelerator.device)
check_rotary(model)
inspect_rotary(model)
# model = accelerator.prepare(model)
# Use 8-bit Adam for trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"[Rank {rank}] parameters catalogued on {accelerator.device}")
optimizer = Adam8bit(trainable_params, lr=learning_rate)
print(f"[Rank {rank}] optimizer built on {accelerator.device}")
# optimizer = accelerator.prepare(optimizer)
# Use this if you are initializing your own optimizer – due to funky 8bit stuff I am leaving it to SFTTrainer

print(f"[Rank {rank}] model & optimizer ready on {accelerator.device}")


## Uncomment these lines to check loading for Deepspeed
# for d in range(torch.cuda.device_count()):
#             print(f"--- Device {d} ---")
#             print(torch.cuda.memory_summary(f"cuda:{d}"))

# Uncomment these lines to check loading for DDP
local_rank = accelerator.local_process_index  # 0, 1, 2, or 3
device = accelerator.device                   # torch.device("cuda:0") for rank0, "cuda:1" for rank1, etc.
print(f"Rank {local_rank} → Device {device}")
print(torch.cuda.memory_summary(device))


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
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
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


# Initialize the SFT trainer with train and eval datasets
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_eval,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    data_collator=default_data_collator,
    optimizers=(optimizer, None),  # use 8-bit Adam
)
inspect_rotary(model)
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