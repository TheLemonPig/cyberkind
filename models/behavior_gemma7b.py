import copy
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

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
            # clone the int8-backed weight to CPU, cast into FP16, then copy
            w = src.weight.data.detach().cpu().clone().to(torch.float16)
            layer.weight.data.copy_(w)
            # same for bias if present
            if src.bias is not None:
                b = src.bias.data.detach().cpu().clone().to(torch.float16)
                layer.bias.data.copy_(b)
            setattr(self, name, layer)

        # infer head count and dimension
        hidden = self.q_proj.out_features
        num_heads = getattr(self_attn, 'num_heads', None) or getattr(self_attn, 'n_heads', None)
        if num_heads and hidden % int(num_heads) == 0:
            self.num_heads = int(num_heads)
        else:
            # fallback to single head
            self.num_heads = 16  # I looked it up, Gemma uses 16 heads
            print(f"Warning: {self_attn.__class__.__name__} has no num_heads or n_heads, assuming 16 heads.")
        self.head_dim = hidden // self.num_heads
        self.scale = self.head_dim ** -0.5
        # optional rotary embeddings
        # self.rotary = 
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
        else:
            assert "rotary is missing"
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
class HybridAttention(nn.Module):
    """
    Split-head attention:
      • first `n_self` heads attend to (queries = x_q, keys/values = x_q)  ➜ self‑attn
      • remaining heads attend to (queries = x_q, keys/values = x_kv)     ➜ cross‑attn

    All projection matrices are cloned from Gemma so logits at t=0 remain identical.
    """
    def __init__(self, orig_attn: nn.Module, n_self: int, mirror: bool = False):
        super().__init__()
        hidden = orig_attn.q_proj.out_features
        self.n_self = n_self
        self.n_total = (
            getattr(orig_attn, "num_heads", None)
            or getattr(orig_attn, "n_heads", None)
            or 16  # default
        )
        if mirror:
            # swap halves: self heads are the last n_self heads instead of first
            self.self_index_start = self.n_total - n_self
        else:
            self.self_index_start = 0
        self.head_dim = hidden // self.n_total

        # Clone shared Q projection (was missing)
        self.q_proj = copy.deepcopy(orig_attn.q_proj).to(torch.float32)

        # Clone shared Q and O
        # separate output projections for self‑ vs cross‑heads (identical at t=0)
        self.o_proj_self  = copy.deepcopy(orig_attn.o_proj).to(torch.float32)
        self.o_proj_cross = copy.deepcopy(orig_attn.o_proj).to(torch.float32)
        self.o_proj = copy.deepcopy(orig_attn.o_proj).to(torch.float32)
        # Clone K/V for self and cross separately
        self.k_self = copy.deepcopy(orig_attn.k_proj).to(torch.float32)
        self.v_self = copy.deepcopy(orig_attn.v_proj).to(torch.float32)
        self.k_cross = copy.deepcopy(orig_attn.k_proj).to(torch.float32)
        self.v_cross = copy.deepcopy(orig_attn.v_proj).to(torch.float32)
        # Rotary + dropout
        # self.rotary  = 
        drop = getattr(orig_attn, 'dropout', None)
        self.dropout = nn.Dropout(drop.p if drop is not None else 0.0)
        self.scale = self.head_dim ** -0.5

    def _reshape(self, x, b):  # (B,T,Hd) ➜ (B,H,T,d)
        return x.view(b, -1, self.n_total, self.head_dim).transpose(1, 2)

    def forward(self, x_q, x_kv, mask=None):
        print("||x_q||_inf", x_q.abs().max().item(), "||x_kv||_inf", x_kv.abs().max().item())
        assert not torch.isnan(x_kv).any(), "NaNs already in x_kv"
        assert not torch.isnan(x_q).any(), "NaNs already in x_q"
        B = x_q.size(0)
        assert not torch.isnan(x_q).any(), "NaNs already in B"
        q = self._reshape(self.q_proj(x_q), B)                     # (B,H,T,d)
        k_self = self._reshape(self.k_self(x_q.to(torch.float32)).to(torch.float16), B)
        v_self = self._reshape(self.v_self(x_q.to(torch.float32)).to(torch.float16), B)
        k_cross = self._reshape(self.k_cross(x_kv.to(torch.float32)).to(torch.float16), B)
        v_cross = self._reshape(self.v_cross(x_kv.to(torch.float32)).to(torch.float16), B)
        assert torch.allclose(
            self.k_self.weight.float(), self.k_cross.weight.float(), atol=0, rtol=0
        ), "Weights diverged!"

        print("any NaN in k_cross.weight? ", torch.isnan(self.k_cross.weight).any())
        print("any NaN in k_cross? ", torch.isnan(self.k_cross).any())
        assert not torch.isnan(k_cross).any(), "NaN caused by creating k_cross"

        i0 = self.self_index_start
        i1 = i0 + self.n_self
        q_self  = q[:,  i0:i1]
        k_s     = k_self[:, i0:i1]
        v_s     = v_self[:, i0:i1]

        # cross heads are the complement
        q_cross = torch.cat([q[:, :i0],  q[:, i1:]], dim=1)
        k_c     = torch.cat([k_cross[:, :i0], k_cross[:, i1:]], dim=1)
        v_c     = torch.cat([v_cross[:, :i0], v_cross[:, i1:]], dim=1)

        assert not torch.isnan(k_c).any(), "NaN caused by cat operation"

        if self.rotary is not None:
            position_ids = None                     # keep default slice
            cos, sin = self.rotary(q_self, position_ids)     # ✅ pass tensor
            q_self,  k_s = apply_rotary_pos_emb(q_self,  k_s, cos, sin, position_ids)
            q_cross, k_c = apply_rotary_pos_emb(q_cross, k_c, cos, sin, position_ids)
        else:
            assert "rotary is missing"
        assert not torch.isnan(k_c).any(), "NaN caused by applying rotary embeddings"
            
        def attn_block(qh, kh, vh):
            # qh, kh, vh are already FP16 on GPU
            #assert not torch.isnan(qh).any(),  "NaN in attention weights at 0"
            #assert not torch.isnan(kh).any(),  "NaN in attention weights at 0.5"
            w = (qh.to(torch.float32) @ kh.to(torch.float32).transpose(-2, -1))  # FP32 matmul
            #assert not torch.isnan(w).any(),  "NaN in attention weights at 1"
            w = w * self.scale
            #assert not torch.isnan(w).any(),  "NaN in attention weights at 2"
            if mask is not None:
                w = w + mask
            #assert not torch.isnan(w).any(),  "NaN in attention weights at 3"
            w = torch.softmax(w, dim=-1).to(torch.float16)     
            #assert not torch.isnan(w).any(),  "NaN in attention weights at 4"                  # down-cast AFTER softmax
            w = self.dropout(w)
            #assert not torch.isnan(w).any(),  "NaN in attention weights at 5"
            out = w @ vh                                                         # FP16 × FP16 OK
            return out

        out_self  = attn_block(q_self,  k_s, v_s)
        assert not torch.isnan(out_self).any(), "NaN after self attn_block"
        out_cross = attn_block(q_cross, k_c, v_c)
        assert not torch.isnan(out_cross).any(), "NaN after cross attn_block"
        out = torch.cat([out_self, out_cross], dim=1)              # (B,H,T,d)
        out_flat = out.transpose(1, 2).reshape(B, -1, self.n_total * self.head_dim)
        out_final = 0.5 * (self.o_proj_self(out_flat) + self.o_proj_cross(out_flat))
        # print("[DBG Hybrid] out_flat", out_flat.shape, "→ out_final", out_final.shape)
        return out_final

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
        out = torch.sigmoid(self.g(x)) * self.w(x)
        print(f"[DBG GatedHighway] x {x.shape} {x.dtype}; w.weight {self.w.weight.shape}")
        return out

# ---------------------------------------------------------------------------
class IdentitySelfAttn(nn.Module):
    """
    Pass-through replacement for Gemma’s SelfAttention.
    Returns (hidden_states, None) while accepting the usual kwargs.
    """
    def forward(
        self,
        hidden_states,
        **kwargs
    ):
        return hidden_states, None  # pass-through, no attention

# ---------------------------------------------------------------------------
class ModuleBlock(nn.Module):
    """Module block: cross-attn + driver + FFN + feedback highway."""
    def __init__(self, src_block: nn.Module, in_dim: int):
        super().__init__()
        # retain original self-attn for cloning
        orig_attn = src_block.self_attn
        # infer hidden dim from projection
        hidden = orig_attn.o_proj.out_features   # 3072 for Gemma-7B
        # print(f"[DBG Init] q_proj weight {orig_attn.q_proj.weight.shape}  hidden={hidden}")
        # print(f"[DBG Init] block hidden={hidden}")

        self.block = src_block
        self.block.self_attn = IdentitySelfAttn()
        # robust head count retrieval
        total_heads = (
            getattr(orig_attn, "num_heads", None)
            or getattr(orig_attn, "n_heads", None)
            or 16  # Gemma default
        )
        # hybrid: 8 self heads + 8 cross heads
        n_self = total_heads // 2

        # hybrid attention: first n_self heads = self, rest = cross
        self.hybrid_attn = HybridAttention(orig_attn, n_self=n_self, mirror=False)
        self.cross_ln = nn.RMSNorm(hidden, eps=1e-5)
        # highways
        self.driver = GatedHighway(hidden)

        # High‑bandwidth feedback: full‑rank Linear + tanh‑bounded scalar gate.
        #   • weight zero‑init  → delta = 0 at t0
        #   • gate starts at 0  → tanh(0)=0 so path closed
        self.w_fb = nn.Linear(hidden, hidden, bias=False, dtype=torch.float16).to(torch.float16)
        nn.init.zeros_(self.w_fb.weight)
        self.gate_fb = nn.Parameter(torch.zeros(1, dtype=torch.float16))   # trainable scalar

    def forward(self, x_mod: torch.Tensor, x_back: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Standard block (4096‑d throughout): hybrid attention → driver → FFN → feedback.
        """
        # Attention (4096‑d everywhere)
        assert not torch.isnan(x_back).any(), "NaN already in x_back"
        attn_out = self.hybrid_attn(x_mod, x_back, mask)
        assert not torch.isnan(attn_out).any(), "NaN before cross_ln"
        attn_out = self.cross_ln(attn_out)

        # Driver (4096‑d)
        drv = self.driver(x_back)

        # Residual combine
        x_comb = x_mod + attn_out + drv

        # Feedback (scalar‑gated)
        delta = torch.tanh(self.gate_fb) * self.w_fb(x_comb)

        # Feed‑forward & norm inside the cloned backbone block
        (x_mod,) = self.block(x_comb, attention_mask=mask, output_attentions=False)
        return x_mod, delta

# ---------------------------------------------------------------------------
class GemmaModular(nn.Module):
    def __init__(self, base: AutoModelForCausalLM):
        super().__init__()
        self.config = base.config
        N = base.config.num_hidden_layers
        self.split = N * 2 // 3
        hidden_dim = base.config.hidden_size   # Gemma‑7B = 4096
        # for i, layer in enumerate(base.model.layers):
        #     print(i, getattr(layer.self_attn.q_proj, "out_features", None))
        # Ensure we start cloning at the first layer whose *weight* rows equal hidden_dim.
        first_4096 = None
        for i, layer in enumerate(base.model.layers):
            out_dim = layer.self_attn.q_proj.weight.shape[0]  # robust (works for Int8Linear)
            if out_dim == hidden_dim:
                first_4096 = i
                break
        # if first_4096 is None:        # fallback: assume every layer is already hidden_dim
        #     first_4096 = 0
        # # print(f'first_4096 layer: {first_4096}, split before clamp: {self.split}')
        # if self.split < first_4096:    # clamp split forward if needed
        #     self.split = first_4096
        # print(f'split final: {self.split}')
        # freeze backbone
        self.backbone_layers = base.model.layers

        for p in self.backbone_layers.parameters():
            p.requires_grad = False
        # ✱A — one shared RoPE helper (lives on the same GPU as the first layer)
        self.rotary_emb = GemmaRotaryEmbedding(self.config).to(
            next(self.backbone_layers.parameters()).device
        )
        self.embed = base.model.embed_tokens; self.embed.requires_grad_(False)
        num_tokens = self.embed.num_embeddings
        self.embed_delta = nn.Embedding(num_tokens, hidden_dim, dtype=torch.float16).to(torch.float16)
        self.delta_gate = nn.Parameter(torch.zeros(1, dtype=torch.float16))
        nn.init.zeros_(self.embed_delta.weight)
        self.pos = getattr(base.model, 'embed_positions', None)
        if self.pos is not None:
            self.pos.requires_grad_(False)
        # -------------------------------------------------------------------
        # build module

        # build module blocks without spiking GPU RAM by deep-copying on CPU first
        self.mod_layers = nn.ModuleList()
        # First ModuleBlock now starts at the model hidden size (4096)
        prev_hidden = hidden_dim
        for bl in self.backbone_layers[self.split:]:
            device = next(bl.parameters()).device
            bl_cpu = bl.to('cpu')
            bl_copy = copy.deepcopy(bl_cpu)
            bl.to(device)

            hidden_curr = bl_copy.self_attn.q_proj.out_features
            mod_block = ModuleBlock(bl_copy, in_dim=prev_hidden)  # map prev → current
            # keep the module in FP16 so its Linear layers match FP16 activations
            mod_block.to(device, dtype=torch.float16)
            self.mod_layers.append(mod_block)

            prev_hidden = hidden_curr  # next block's "in_dim"

        self.ln_f = copy.deepcopy(base.model.norm)
        self.lm_head = copy.deepcopy(base.lm_head)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        if attention_mask is not None and attention_mask.dtype == torch.long:
            attention_mask = attention_mask.to(dtype=torch.bool)
        B, T = input_ids.shape
        h_back = self.embed(input_ids)
        if self.pos is not None:
            h_back = h_back + self.pos(torch.arange(T, device=h_back.device))[None, :]
        assert not torch.isnan(h_back).any(), "NaN already in h_back from very beginning"
        # frozen until split
        # build RoPE tuple for this sequence
        seq_len = h_back.size(1)
        position_ids = torch.arange(seq_len, device=h_back.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(h_back, position_ids)
        for idx, layer in enumerate(self.backbone_layers[:self.split]):
            assert not torch.isnan(h_back).any(), f"NaN already in h_back {idx} layers in"
            h_back = layer(
                h_back,
                position_embeddings=(cos, sin),             # ← NEW
                attention_mask=attention_mask,
                output_attentions=False,
            )[0]
        # module initial state
        h_mod = h_back.clone()
        feedback = torch.zeros_like(h_back)
        # paired layers
        for back_layer, mod_block in zip(
            self.backbone_layers[self.split:], self.mod_layers
        ):
            h_back = back_layer(
                h_back + feedback,
                position_embeddings=(cos, sin),             # ← NEW
                attention_mask=attention_mask,
                output_attentions=False
            )[0]
            # give the tuple to this block’s HybridAttention
            mod_block.hybrid_attn.rotary = lambda x, pos=None, _cs=(cos, sin): _cs
            h_mod, feedback = mod_block(h_mod, h_back, attention_mask)
            # add tanh-gated delta embedding
            print(h_mod, self.embed_delta(input_ids), self.delta_gate)
            h_mod = h_mod + torch.tanh(self.delta_gate) * self.embed_delta(input_ids)
        logits = self.lm_head(self.ln_f(h_mod))

        # When SFTTrainer passes labels, compute causal‑LM loss
        if labels is not None:
            # shift logits and labels for causal‑LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return {"loss": loss, "logits": logits}

        # inference / generate mode
        return {"logits": logits}

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
