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

PREDICTION_ID = "google/gemma-7b"
BEHAVIOR_ID   = "google/gemma-7b"
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
    PREDICTION_ID,
    token=hf_token
    )
EOS_TOKEN = tokenizer.eos_token

if load_in_8bit:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float32,   # safer accumulator
        #bnb_8bit_compute_dtype=torch.float32,   # <- critical change
        llm_int8_threshold=6.0,            # optional tuning threshold
        # llm_int8_has_fp16_weight=False,    # optional
    )
    rank = accelerator.local_process_index
    print(f"[Rank {rank}] about to load backbone on GPU {torch.cuda.current_device()}")
    print(f"[Rank {rank}/{world_size}] 📦 loading model now…", flush=True)
    predict = AutoModelForCausalLM.from_pretrained(
        PREDICTION_ID,
        quantization_config=quant_config,
        # device_map=accelerator.device, #'auto',   # Uncomment this unless you are using DDP
        output_hidden_states=True,
        token=hf_token,
    )
    behave = AutoModelForCausalLM.from_pretrained(
        BEHAVIOR_ID,
        quantization_config=quant_config,
        # device_map=accelerator.device, #'auto',   # Uncomment this unless you are using DDP
        output_hidden_states=True,
        token=hf_token,
    )
    print(f"[Rank {rank}/{world_size}] 📦 done loading model", flush=True)
    print(f"[Rank {rank}] finished loading backbone")
else:
    predict = AutoModelForCausalLM.from_pretrained(
        PREDICTION_ID,
        torch_dtype=torch.bfloat16,
        # device_map=accelerator.device,  # Uncomment this unless you are using DDP
        output_hidden_states=True,
        token=hf_token,
    )
    behave = AutoModelForCausalLM.from_pretrained(
        BEHAVIOR_ID,
        torch_dtype=torch.bfloat16,
        # device_map=accelerator.device,  # Uncomment this unless you are using DDP
        output_hidden_states=True,
        token=hf_token,
    )
# -----------------------------
torch.autograd.set_detect_anomaly(True)
model = GemmaModular(predict, behave)
#torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# def _cast_fp16(_, __, output):
#     return output.to(torch.float16)

# for layer in model.backbone_layers:
#     attn = layer.self_attn
#     attn.q_proj.register_forward_hook(_cast_fp16)
#     attn.k_proj.register_forward_hook(_cast_fp16)
#     attn.v_proj.register_forward_hook(_cast_fp16)

# ---------------------------------------------------------------------------
#  LIGHTWEIGHT DEBUG HOOKS – will pin‑point first inf/nan without altering the
#  model's forward signatures.
# ---------------------------------------------------------------------------
# def add_debug_hooks(model):
#     """
#     Register forward hooks on every Linear layer and on attention
#     score tensors.  Stops in pdb at the first inf/nan.
#     """
#     import types, pdb

#     def stats(name, t):
#         if not torch.is_tensor(t):
#             return
#         with torch.no_grad():
#             mx, mn = t.max().item(), t.min().item()
#             bad = torch.isinf(t).any() or torch.isnan(t).any()
#         print(f"{name:<45}  [{mn:8.2f}, {mx:8.2f}]" + ("  <-- 🔥" if bad else ""))
#         if bad:
#             pdb.set_trace()

#     # 1) every Linear's output
#     for n, m in model.named_modules():
#         if isinstance(m, nn.Linear):
#             m.register_forward_hook(lambda mod, _, out, n=n: stats(f"{n}.out", out))

#     # 2) hidden‑state input to each GemmaAttention (before qkv projections)
#     from transformers.models.gemma.modeling_gemma import GemmaAttention
#     from functools import partial

#     def attn_pre_hook(name, stats_fn, module, args, kwargs):
#         """
#         Works for both positional and keyword calling conventions.
#         Triggered immediately before q/k/v projection.
#         """
#         hidden = args[0] if args else kwargs.get("hidden_states", None)
#         if hidden is not None:
#             stats_fn(f"{name}.attn_in", hidden)

#     for name, module in model.named_modules():
#         if isinstance(module, GemmaAttention) and not hasattr(module, "_dbg"):
#             hook = partial(attn_pre_hook, name, stats)  # capture layer name & stats fn
#             module.register_forward_pre_hook(hook, with_kwargs=True)
#             module._dbg = True  # avoid double‑hooking

# # attach hooks once
# add_debug_hooks(model)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  EXTRA SPY: print layer‑norm and q/k/v outputs for decoder layer 18 only
# ---------------------------------------------------------------------------
# def add_layer18_spy(model):
#     """
#     For backbone layer 18, print stats right after layer‑norm and after each
#     q/k/v projection, then abort the run so you don’t hit the overflow.
#     """
#     import math, sys

#     try:
#         dec18 = model.backbone_layers[18]          # GemmaDecoderLayer
#         attn18 = dec18.self_attn                   # GemmaAttention inside it
#     except (AttributeError, IndexError):
#         print("⚠️  Could not locate backbone_layers[18]; spy not installed.")
#         return

#     def spy(module, args, kwargs):
#         # hidden_states may be positional or keyword
#         h = args[0] if args else kwargs.get("hidden_states")
#         ln = dec18.input_layernorm   # RMSNorm lives in the decoder layer
#         x  = ln(h)

#         print("\n=== L18 SPY ===")
#         print("after layer‑norm      ", x.abs().min().item(), x.abs().max().item())

#         for tag, proj in {"q": module.q_proj,
#                           "k": module.k_proj,
#                           "v": module.v_proj}.items():
#             # --- put this inside the for-loop over {"q":…, "k":…, "v":…} --------------
#             out = proj(x)
#             print(f"{tag}_proj out        [{out.abs().min():.5f}, {out.abs().max():.5f}]")

#             # ---- try every known scale/zero layout -----------------------------------
#             scale_t, zero_t = None, None
#             if hasattr(proj, "scales"):                # classic layout (≤ 0.41)
#                 scale_t, zero_t = proj.scales, proj.zeros
#             elif hasattr(proj.weight, "SCB"):          # v0.42-0.44
#                 scale_t, zero_t = proj.weight.SCB[:, 0], proj.weight.SCB[:, 1]
#             elif hasattr(proj.weight, "CB"):           # ≥ 0.45
#                 scale_t, zero_t = proj.weight.CB[:, 0], proj.weight.CB[:, 1]

#             if scale_t is not None:
#                 print(f"  scale  min/max      {scale_t.min().item():.3e}  "
#                     f"{scale_t.max().item():.3e}")
#                 print(f"  zero   min/max      {zero_t.min().item():.0f}     "
#                     f"{zero_t.max().item():.0f}")
#             else:
#                 print("  ⚠️  scales/zeros attribute not found on this BnB build")

#             # ---- de-quantise weights once to see the *real* fp32 matrix --------------
#             w_int8 = proj.weight       # raw int8
#             if scale_t is not None:
#                 w_fp32 = (w_int8.float() - zero_t.unsqueeze(1)) * scale_t.unsqueeze(1)
#             else:
#                 w_fp32 = w_int8.float()   # fall-back (won’t match real kernel)

#             print(f"  weight abs-max       {w_fp32.abs().max():.3e}")
#             print(f"  weight inf/nan       inf={torch.isinf(w_fp32).any()}  "
#                 f"nan={torch.isnan(w_fp32).any()}")
#             # --------------------------------------------------------------------------

#             # — de‑quantised weight stats —
#             w = proj.weight.float()          # convert 8‑bit block to fp32
#             print(f"  weight abs‑max       {w.abs().max():.3e}")
#             print(f"  weight inf/nan       inf={torch.isinf(w).any()}  nan={torch.isnan(w).any()}")

#         # stop after printing once so logs stay short
#         sys.exit(0)

#     # `prepend=True` ensures the spy runs *before* original forward
#     attn18.register_forward_pre_hook(spy, with_kwargs=True, prepend=True)
#     print("🔍 Installed layer‑18 spy hook")

# # Attach the spy once
# # Attach the spy once
# add_layer18_spy(model)
# # ---------------------------------------------------------------------------
# #  EXTRA SPY 2: print min/max of Q and K right *after* rotary embedding
# # ---------------------------------------------------------------------------
# from functools import partial
# from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

# def rot_hook(tag, module, args, output):
#     # output is (q_after, k_after)
#     q_after, k_after = output
#     print(f"{tag}.after_rotary  q  [{q_after.abs().min():.2f}, {q_after.abs().max():.2f}]")
#     print(f"{tag}.after_rotary  k  [{k_after.abs().min():.2f}, {k_after.abs().max():.2f}]")

# try:
#     l18_attn = model.backbone_layers[18].self_attn
#     # Gemma’s attention module exposes .rotary
#     l18_attn.rotary.register_forward_hook(partial(rot_hook, "L18"), with_kwargs=False)
#     print("🔍 Installed rotary spy for layer‑18")
# except (AttributeError, IndexError):
#     print("⚠️  Could not install rotary spy – layer 18 missing.")
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

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
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
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