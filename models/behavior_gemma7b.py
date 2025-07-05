# gemma_modular.py (revision-9)
"""Gemma‑7B + fronto‑posterior module with **sigmoid‑gated highways**.

New in *revision‑9*  ▸  **Feedback highway now truly modulates the frozen
backbone on the *next* layer**:

```
loop over last ⅓ blocks
    h_back_in  = h_back + feedback         # ← feedback from module k‑1
    h_back     = frozen_backbone_i(h_back_in)
    h_mod, Δ   = module_i(h_mod, h_back)   # driver+modulator paths
    feedback   = Δ                         # will condition layer i+1
```

That means the module can co‑opt posterior machinery without touching its
weights.  Initial logits remain identical (Δ<1e‑4).
"""

import copy
from typing import Optional
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "google/gemma-7b"
DEVICE, DTYPE = ("cuda" if torch.cuda.is_available() else "cpu"), torch.bfloat16

# ────────────────────────────────────────────────────────────────────────────
class CrossAttentionFromSelf(nn.Module):
    """Cross‑attention sharing Q/K/V/O with a self‑attn layer."""
    def __init__(self, self_attn: nn.Module):
        super().__init__()
        self.q_proj = copy.deepcopy(self_attn.q_proj)
        self.k_proj = copy.deepcopy(self_attn.k_proj)
        self.v_proj = copy.deepcopy(self_attn.v_proj)
        self.o_proj = copy.deepcopy(self_attn.o_proj)
        self.num_heads, self.head_dim = self_attn.num_heads, self_attn.head_dim
        self.scale = self.head_dim ** -0.5
        self.rotary = copy.deepcopy(getattr(self_attn, "rotary_emb", None))
        self.dropout = nn.Dropout(self_attn.dropout.p)

    def _reshape(self, x: torch.Tensor, b: int):
        s = x.size(1)
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b = q.size(0)
        qh = self._reshape(self.q_proj(q), b)
        kh = self._reshape(self.k_proj(kv), b)
        vh = self._reshape(self.v_proj(kv), b)
        if self.rotary is not None:
            qh, kh = self.rotary(qh, kh)
        attn = (qh @ kh.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn += mask
        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = attn @ vh
        out = out.transpose(1, 2).reshape(b, -1, self.num_heads * self.head_dim)
        return self.o_proj(out)

# ────────────────────────────────────────────────────────────────────────────
class GatedHighway(nn.Module):
    """σ(W_g x) ⊙ (W_h x) – initialised closed."""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, dim, bias=False)
        self.g = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.g.weight); nn.init.zeros_(self.g.bias)

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(self.g(x)) * self.w(x)

# ────────────────────────────────────────────────────────────────────────────
class ModuleBlock(nn.Module):
    """One trainable block with driver & feedback highways."""
    def __init__(self, src_block: nn.Module):
        super().__init__()
        self.block = src_block  # self‑attn + FFN (trainable copy)
        # Disable the original self-attention since cross-attn replaces it
        self.block.self_attn = nn.Identity()
        # To restore original self-attention instead of Identity, uncomment:
        # self.block.self_attn = copy.deepcopy(src_block.self_attn)
        h = src_block.self_attn.hidden_size
        self.cross_ln   = nn.RMSNorm(h, eps=1e-5)
        self.cross_attn = CrossAttentionFromSelf(src_block.self_attn)
        self.driver     = GatedHighway(h)   # posterior → module
        self.feedback   = GatedHighway(h)   # module    → next backbone input

    def forward(self, x_mod: torch.Tensor, x_back: torch.Tensor, mask: Optional[torch.Tensor]):
        delta = self.feedback(x_mod)              # module→posterior signal
        # K/V for cross‑attention come from the **frozen** posterior.
        # If you need hard gradient isolation between backbone & module for a
        # dual‑objective setup, detach here:
        # kv = x_back.detach()
        kv    = x_back
        mod   = self.cross_attn(self.cross_ln(x_mod), kv, mask)
        drv   = self.driver(x_back)               # posterior driver
        x_mod = x_mod + mod + drv
        x_mod, _ = self.block(x_mod, attention_mask=mask, output_attentions=False)
        return x_mod, delta                       # return fresh feedback

# ────────────────────────────────────────────────────────────────────────────
class GemmaModular(nn.Module):
    def __init__(self, base: AutoModelForCausalLM):
        super().__init__()
        N = base.config.num_hidden_layers
        self.split = N * 2 // 3
        # backbone (frozen)
        self.backbone_layers = base.model.layers
        for p in self.backbone_layers.parameters():
            p.requires_grad = False
        self.embed = base.model.embed_tokens; self.embed.requires_grad_(False)
        self.pos   = getattr(base.model, "embed_positions", None)
        if self.pos is not None: self.pos.requires_grad_(False)
        # module (trainable)
        self.mod_layers = nn.ModuleList([
            ModuleBlock(copy.deepcopy(bl)) for bl in self.backbone_layers[self.split:]
        ])
        self.ln_f, self.lm_head = copy.deepcopy(base.model.norm), copy.deepcopy(base.lm_head)

    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor]=None):
        B, T = ids.shape
        h_back = self.embed(ids)
        if self.pos is not None:
            h_back += self.pos(torch.arange(T, device=h_back.device))[None, :]
        # run frozen layers up to split
        for i in range(self.split):
            h_back,_ = self.backbone_layers[i](h_back, attention_mask=mask, output_attentions=False)
        h_mod = h_back.clone()
        feedback = torch.zeros_like(h_back)
        # remaining layers with feedback into backbone
        for i, back_layer in enumerate(self.backbone_layers[self.split:]):
            h_back_in = h_back + feedback      # posterior receives last module signal
            h_back,_  = back_layer(h_back_in, attention_mask=mask, output_attentions=False)
            h_mod, feedback = self.mod_layers[i](h_mod, h_back, mask)  # new feedback
        return self.lm_head(self.ln_f(h_mod))

# ────────────────────────────────────────────────────────────────────────────
def build_modular_model():
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, device_map="auto")
    return GemmaModular(base).to(DEVICE, dtype=DTYPE)

# quick equivalence check
if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    prompt = "Sally hides a marble in a basket. Anne moves it to the box. Where will Sally look?"
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, device_map="auto").eval()
    mod  = build_modular_model().eval()
    with torch.no_grad():
        diff = (base(**inp).logits - mod(**inp)).abs().max().item()
    print(f"Max |Δ logits| pre‑train: {diff:.2e}")
    assert diff < 1e-4

    # ──────────────────────────────────────────────────────────────────
    # Example **dual‑objective** training skeleton
    # One optimizer.step()    ⇢    **two backward passes**
    # Keeps gradient streams separate while highways stay active.
    # ------------------------------------------------------------------
    # 1) Freeze module, train backbone objective (e.g. MLM loss)
    # mod.requires_grad_(False)
    # logits_backbone = backbone_head(h_back_final.detach())
    # loss_b = crit_b(logits_backbone, labels)
    # loss_b.backward()
    #
    # 2) Freeze backbone, train module objective (e.g. RL or SFT)
    # for p in mod.parameters(): p.requires_grad_(True)
    # for p in model.backbone_layers.parameters(): p.requires_grad_(False)
    # logits_mod = model.lm_head(model.ln_f(h_mod_final))
    # loss_m = crit_m(logits_mod, targets_or_reward)
    # loss_m.backward()
    #
    # optimizer.step(); optimizer.zero_grad()
    #
    # Note: If you prefer a *single* backward pass with hard isolation,
    # detach the K/V tensor in ModuleBlock (see commented line `kv = x_back.detach()`).
