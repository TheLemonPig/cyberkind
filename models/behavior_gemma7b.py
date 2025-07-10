# gemma_modular.py (final revision)
"""Gemma‑7B with fronto-posterior module and sigmoid-gated highways.

Design recap:
• Frozen backbone: all original Gemma blocks 0 … N-1
• Trainable module: deep‑copied last N/3 blocks (blocks split…N-1)

Each module block performs:
  1. Cross-attention (module queries, backbone keys/values)
  2. Additive gated driver from backbone
  3. Residual update
  4. Feed-forward (copied FFN) — original self-attn disabled by default
  5. Feedback gated highway output to next backbone input

Highways use σ(W_g x) ⊙ (W_h x), zero‑initialized to start inert.
"""

import copy
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "google/gemma-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
class CrossAttentionFromSelf(nn.Module):
    """Cross-attention reusing an existing self-attention's Q/K/V/O weights."""
    def __init__(self, self_attn: nn.Module):
        super().__init__()
        # Recreate the 4 projection layers from the original, cloning weights on CPU
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            src = getattr(self_attn, name)
            # build a fresh Linear with the same dims, on CPU
            layer = nn.Linear(src.in_features, src.out_features, bias=src.bias is not None)
            # clone the int8-backed weight to CPU, cast into our working dtype, then copy
            w = src.weight.data.detach().cpu().clone().to(DTYPE)
            layer.weight.data.copy_(w)
            # same for bias if present
            if src.bias is not None:
                b = src.bias.data.detach().cpu().clone().to(DTYPE)
                layer.bias.data.copy_(b)
            setattr(self, name, layer)

        # infer head count and dimension
        hidden = self.q_proj.out_features
        num_heads = getattr(self_attn, 'num_heads', None) or getattr(self_attn, 'n_heads', None)
        if num_heads and hidden % int(num_heads) == 0:
            self.num_heads = int(num_heads)
            self.head_dim = hidden // self.num_heads
        else:
            # fallback to single head
            self.num_heads = 1
            self.head_dim = hidden
        self.scale = self.head_dim ** -0.5
        # optional rotary embeddings
        self.rotary = copy.deepcopy(getattr(self_attn, 'rotary_emb', None))
        # dropout
        drop = getattr(self_attn, 'dropout', None)
        self.dropout = nn.Dropout(drop.p if drop is not None else 0.0)

    def _reshape(self, x: torch.Tensor, batch: int):
        seq = x.size(1)
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz = query.size(0)
        # project and reshape
        q = self._reshape(self.q_proj(query), bsz)
        k = self._reshape(self.k_proj(key_value), bsz)
        v = self._reshape(self.v_proj(key_value), bsz)
        # rotary if present
        if self.rotary is not None:
            q, k = self.rotary(q, k)
        # scaled dot-product
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # combine heads
        out = attn @ v  # (bsz, heads, seq, head_dim)
        out = out.transpose(1, 2).reshape(bsz, -1, self.num_heads * self.head_dim)
        return self.o_proj(out)

# ---------------------------------------------------------------------------
class GatedHighway(nn.Module):
    """σ(W_g x) ⊙ (W_h x), zero‑initialized to produce 0 at start."""
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, dim, bias=False)
        self.g = nn.Linear(dim, dim, bias=True)
        # zero init
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.g.weight); nn.init.zeros_(self.g.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.g(x)) * self.w(x)

# ---------------------------------------------------------------------------
class ModuleBlock(nn.Module):
    """Module block: cross-attn + driver + FFN + feedback highway."""
    def __init__(self, src_block: nn.Module):
        super().__init__()
        # retain original self-attn for cloning
        orig_attn = src_block.self_attn
        # infer hidden dim from projection
        hidden = orig_attn.q_proj.out_features
        # copy block and disable its self-attn
        # self.block = copy.deepcopy(src_block)
        # src_block has already been deep-copied on the CPU,
        # so just take it as-is and disable its self-attn
        self.block = src_block
        self.block.self_attn = nn.Identity()
        # cross-attention
        # for d in range(torch.cuda.device_count()):
        #     print(f"--- Device {d} ---")
        #     print(torch.cuda.memory_summary(f"cuda:{d}"))
        self.cross_attn = CrossAttentionFromSelf(orig_attn)
        self.cross_ln = nn.RMSNorm(hidden, eps=1e-5)
        # highways
        self.driver = GatedHighway(hidden)
        self.feedback = GatedHighway(hidden)

    def forward(self, x_mod: torch.Tensor, x_back: torch.Tensor, mask: Optional[torch.Tensor]):
        # feedback signal to next backbone layer
        delta = self.feedback(x_mod)
        # modulator: cross-attn
        kv = x_back  # detach if hard isolation needed: x_back.detach()
        mod = self.cross_attn(self.cross_ln(x_mod), kv, mask)
        # driver
        drv = self.driver(x_back)
        # combine
        x_mod = x_mod + mod + drv
        # feed-forward + norm
        x_mod, _ = self.block(x_mod, attention_mask=mask, output_attentions=False)
        return x_mod, delta

# ---------------------------------------------------------------------------
class GemmaModular(nn.Module):
    def __init__(self, base: AutoModelForCausalLM):
        super().__init__()
        N = base.config.num_hidden_layers
        self.split = N * 2 // 3
        # freeze backbone
        self.backbone_layers = base.model.layers
        for p in self.backbone_layers.parameters():
            p.requires_grad = False
        self.embed = base.model.embed_tokens; self.embed.requires_grad_(False)
        self.pos = getattr(base.model, 'embed_positions', None)
        if self.pos is not None:
            self.pos.requires_grad_(False)
        # build module

        # self.mod_layers = nn.ModuleList([
        #     ModuleBlock(copy.deepcopy(bl)) for bl in self.backbone_layers[split:]
        # ])
        # build module blocks without spiking GPU RAM by deep-copying on CPU first
        self.mod_layers = nn.ModuleList()
        for bl in self.backbone_layers[self.split:]:
            # remember which GPU this layer lived on
            device = next(bl.parameters()).device
            # temporarily move the layer to CPU then deep-copy it there
            bl_cpu = bl.to('cpu')
            bl_copy = copy.deepcopy(bl_cpu)
            # move the original layer back to its device
            bl.to(device)
            # initialize our ModuleBlock from the CPU copy, then send it to the right GPU
            mod_block = ModuleBlock(bl_copy)
            mod_block.to(device)
            self.mod_layers.append(mod_block)
        self.ln_f = copy.deepcopy(base.model.norm)
        self.lm_head = copy.deepcopy(base.lm_head)

    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T = ids.shape
        h_back = self.embed(ids)
        if self.pos is not None:
            h_back = h_back + self.pos(torch.arange(T, device=h_back.device))[None, :]
        # frozen until split
        for layer in self.backbone_layers[:self.split]:
            h_back, _ = layer(h_back, attention_mask=mask, output_attentions=False)
        # module initial state
        h_mod = h_back.clone()
        feedback = torch.zeros_like(h_back)
        # paired layers
        for back_layer, mod_block in zip(self.backbone_layers[self.split:], self.mod_layers):
            # backbone with feedback
            h_back = back_layer(h_back + feedback, attention_mask=mask, output_attentions=False)[0]
            # module step
            h_mod, feedback = mod_block(h_mod, h_back, mask)
        return self.lm_head(self.ln_f(h_mod))

# ---------------------------------------------------------------------------
def build_modular_model():
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, device_map='auto')
    return GemmaModular(base).to(DEVICE, dtype=DTYPE)

# quick equivalence & dual-objective template
if __name__ == '__main__':
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    prompt = 'Sally hides a marble...'
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=DTYPE, device_map='auto').eval()
    mod = build_modular_model().eval()
    with torch.no_grad(): diff = (base(**inp).logits - mod(**inp)).abs().max().item()
    print(f'Max |Δ logits| pre-train: {diff:.2e}')
    assert diff < 1e-4

    # dual-objective example:
    # one optimizer.step() with two backward passes
    # 1) freeze mod, train backbone loss: loss_b.backward()
    # 2) freeze backbone, train module loss: loss_m.backward()
    # optimizer.step(); optimizer.zero_grad()
