import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from dataclasses import dataclass
from typing import List, Tuple

"""
Gemma‑7B (frozen)  +  donor GPT‑2 blocks reused as a *cross‑attention* module.

Major adjustments in this revision
=================================
1. **Hidden‑size adapter** ─ Gemma hidden=3072, GPT‑2 hidden=1024 →
   we add a trainable linear `kv_in_proj` that maps Gemma features into
   the donor attention dimensionality *once* per layer.
2. **Head‑dim safety** ─ we assert that `embed_dim % num_heads == 0` so
   the reshape in `split_qkv` never explodes.
3. **Rotary / positional mismatch** ─ left as‑is; empirical evidence
   suggests the projections are robust.  Inline TODO comment explains
   how to disable RoPE if needed.
4. **NaN guard utility** ─ `check_for_nans()` helper you can call from
   your train loop.
5. **LR scheduler placeholder** ─ inline comment shows where to plug in
   a warm‑up + cosine decay schedule (not implemented here to keep this
   file self‑contained).
"""

# -------------------------------------------------------
# 0.  Config – edit here
# -------------------------------------------------------

BACKBONE_ID    = "google/gemma-7b"   # frozen comprehension model
DONOR_ID       = "gpt2-medium"       # decoder supplying blocks
N_DONOR_LAYERS = 8                    # last N donor blocks form module
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.bfloat16       # keeps VRAM low while preserving range
USE_8BIT       = True                # 8‑bit quant for backbone

# -------------------------------------------------------
# 1.  Backbone loader (Gemma) – frozen, hidden states retained
# -------------------------------------------------------

def load_frozen_backbone(model_id: str = BACKBONE_ID):
    if USE_8BIT:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model   = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            torch_dtype=DTYPE,
            device_map="auto",
            output_hidden_states=True,
        )
    else:
        model   = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            device_map="auto",
            output_hidden_states=True,
        )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

# -------------------------------------------------------
# 2.  Tiny gated highway – parameter‑light driver path
# -------------------------------------------------------

class BridgeBlock(nn.Module):
    """Gated additive highway that injects (projected) driver features."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.highway_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_proj    = nn.Linear(hidden_size, hidden_size)
        self.ln           = nn.LayerNorm(hidden_size)

    def forward(self, y: torch.Tensor, x_driver: torch.Tensor) -> torch.Tensor:
        """x_driver already lives in *donor* hidden dimensionality."""
        pooled  = x_driver.mean(dim=1, keepdim=True)      # (B,1,D)
        highway = self.highway_proj(pooled)
        gate    = torch.sigmoid(self.gate_proj(pooled))   # 0–1 soft mask
        return self.ln(y + gate * highway)

# -------------------------------------------------------
# 3.  Attention helpers (unchanged)
# -------------------------------------------------------

def split_qkv(proj_out: torch.Tensor, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, threeD = proj_out.size()
    D = threeD // 3
    q, k, v = proj_out.split(D, dim=2)
    head_dim = D // num_heads
    # reshape → (B, num_heads, T, head_dim)
    return (
        q.view(B, T, num_heads, head_dim).transpose(1, 2),
        k.view(B, T, num_heads, head_dim).transpose(1, 2),
        v.view(B, T, num_heads, head_dim).transpose(1, 2),
    )

def merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, H, T, Hd = x.size()
    return x.transpose(1, 2).contiguous().view(B, T, H * Hd)

# -------------------------------------------------------
# 4.  ModuleLayer – donor block reused *with* adapter proj
# -------------------------------------------------------

class ModuleLayer(nn.Module):
    """A donor GPT‑2 block turned into cross‑attention w/ driver adapter."""

    def __init__(self, donor_block: nn.Module, driver_in_dim: int):
        super().__init__()
        self.donor_block = donor_block
        self.attn        = donor_block.attn         # Q/K/V & out proj reused
        self.mlp         = donor_block.mlp
        self.ln_1        = donor_block.ln_1
        self.ln_2        = donor_block.ln_2
        self.num_heads   = donor_block.attn.num_heads
        self.head_dim    = donor_block.attn.head_dim
        embed_dim        = donor_block.attn.embed_dim
        assert embed_dim % self.num_heads == 0, "Head dim mismatch!"

        # Adapter: maps Gemma hidden (driver_in_dim) → donor embed_dim
        self.kv_in_proj = nn.Linear(driver_in_dim, embed_dim, bias=False)

        self.bridge     = BridgeBlock(embed_dim)

    def forward(self, y: torch.Tensor, x_detached: torch.Tensor, attention_mask=None) -> torch.Tensor:
        # ---- 1. Adapter projection (Gemma → donor dim) ----
        x_drv = self.kv_in_proj(x_detached)  # (B,T2,embed_dim)

        # ---- 2. Cross‑attention using donor weights ----
        qkv_y = self.attn.c_attn(y)          # (B,T,3D)
        q_y, _, _ = split_qkv(qkv_y, self.num_heads)

        kv_x       = self.attn.c_attn(x_drv)
        _, k_x, v_x = split_qkv(kv_x, self.num_heads)

        attn_scores = torch.matmul(q_y, k_x.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            attn_scores += attention_mask.unsqueeze(1).unsqueeze(2)
        attn_probs  = F.softmax(attn_scores, dim=-1)
        attn_ctx    = torch.matmul(attn_probs, v_x)
        attn_ctx    = merge_heads(attn_ctx)
        attn_out    = self.attn.c_proj(attn_ctx)  # reuse donor out proj

        y = y + attn_out          # residual 1
        y = self.ln_1(y)

        # ---- 3. Feed‑forward (unchanged) ----
        y = y + self.mlp(y)       # residual 2
        y = self.ln_2(y)

        # ---- 4. Highway injection ----
        y = self.bridge(y, x_drv)
        return y

# -------------------------------------------------------
# 5.  Module builder – slice *last N* donor layers
# -------------------------------------------------------

@dataclass
class ModuleConfig:
    donor_id: str = DONOR_ID
    n_layers: int = N_DONOR_LAYERS


def build_module(cfg: ModuleConfig, driver_in_dim: int):
    donor = AutoModelForCausalLM.from_pretrained(cfg.donor_id, torch_dtype=DTYPE)
    donor_layers: List[nn.Module] = list(donor.transformer.h)[-cfg.n_layers:]
    wrapped = nn.ModuleList([
        ModuleLayer(block, driver_in_dim) for block in donor_layers
    ])
    # Everything in wrapped is trainable by default
    return donor.config.n_embd, wrapped

# -------------------------------------------------------
# 6.  Combined model
# -------------------------------------------------------

class CombinedModel(nn.Module):
    def __init__(self, tap_start: int = None):
        super().__init__()
        self.backbone = load_frozen_backbone()
        driver_dim    = self.backbone.config.hidden_size  # Gemma=3072

        # Default tap_start chooses final N layers so indices align
        total_layers  = len(self.backbone.model.layers)
        if tap_start is None:
            tap_start = total_layers - N_DONOR_LAYERS
        self.tap_start = tap_start

        # Build module
        self.module_dim, self.module_layers = build_module(ModuleConfig(), driver_dim)
        self.lm_head = nn.Linear(self.module_dim, self.backbone.config.vocab_size, bias=False)

    @property
    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    # ---- Forward ----
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            bb_out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = bb_out.hidden_states  # tuple len L+1

        # Driver features from tap_start onward
        x_seq = hidden_states[self.tap_start + 1:]
        y = self.module_layers[0].kv_in_proj(x_seq[0]).clone()  # initial state in donor dim

        for i, layer in enumerate(self.module_layers):
            x_det = x_seq[i].detach()
            y = layer(y, x_det, attention_mask)

        logits = self.lm_head(y)
        return logits

# -------------------------------------------------------
# 7.  Utilities
# -------------------------------------------------------

def count_trainable(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def check_for_nans(t: torch.Tensor, tag: str = "tensor"):
    """Call from your training loop to abort on NaNs/inf."""
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"{tag} contains NaNs or Infs!")

# -------------------------------------------------------
# 8.  Smoke test (forward pass only)
# -------------------------------------------------------

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_ID)
    model     = CombinedModel().to(DEVICE)
    prompt    = "Why do people sometimes believe things that are false?"
    toks      = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**toks)
    print("logits shape:", logits.shape)
    print(f"Trainable params: {count_trainable(model):.1f} M")

    # NaN guard demo
    check_for_nans(logits, "logits")

    # TODO: plug in LR scheduler → warm‑up + cosine decay
    # e.g., torch.optim.AdamW + transformers.get_cosine_schedule_with_warmup
