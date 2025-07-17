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
        self.q_self = orig_attn.q_proj.to(torch.bfloat16)[:,self.head_dim:]

        # Clone K/V for self and cross separately
        self.k_self = orig_attn.k_proj.to(torch.bfloat16)[:,self.head_dim:]
        self.k_cross = orig_attn.k_proj.to(torch.bfloat16)[:,:self.head_dim]
        self.v_self = orig_attn.v_proj.to(torch.bfloat16)[:,self.head_dim:]
        self.v_cross = orig_attn.v_proj.to(torch.bfloat16)[:,:self.head_dim]
        # Clone shared Q and O
        # separate output projections for self‑ vs cross‑heads (identical at t=0)
        self.o_proj_self  = orig_attn.o_proj.to(torch.bfloat16)[:,self.head_dim:]
        self.o_proj_cross = orig_attn.o_proj.to(torch.bfloat16)[:,:self.head_dim]
        # dropout
        drop = getattr(orig_attn, 'dropout', None)
        self.dropout = nn.Dropout(drop.p if drop is not None else 0.0)
        self.scale = self.head_dim ** -0.5

    def _reshape(self, x, b):  # (B,T,Hd) ➜ (B,H,T,d)
        return x.view(b, -1, self.n_total, self.head_dim).transpose(1, 2)

    def forward(self, x_self, x_cross, mask=None):
        print("||x_self||_inf", x_self.abs().max().item(), "||x_cross||_inf", x_cross.abs().max().item())
        assert not torch.isnan(x_cross).any(), "NaNs already in x_kv"
        assert not torch.isnan(x_self).any(), "NaNs already in x_q"
        B = x_self.size(0)
        assert not torch.isnan(x_self).any(), "NaNs already in B"
        q_cross = self._reshape(self.q_self(x_cross), B)
        q_self = self._reshape(self.q_self(x_self), B)                     # (B,H,T,d)
        k_self = self._reshape(self.k_self(x_self), B)
        v_self = self._reshape(self.v_self(x_self), B)
        k_cross = self._reshape(self.k_cross(x_self), B)
        v_cross = self._reshape(self.v_cross(x_self), B)
        assert torch.allclose(
            self.k_self.weight.float(), self.k_cross.weight.float(), atol=0, rtol=0
        ), "Weights diverged!"

        print("any NaN in k_cross.weight? ", torch.isnan(self.k_cross.weight).any())
        assert not torch.isnan(k_cross).any(), "NaN caused by creating k_cross"
            
        def attn_block(qh, kh, vh):
            # qh, kh, vh are already BF16 on GPU
            w = (qh.to(torch.float32) @ kh.to(torch.float32).transpose(-2, -1)) * self.scale
            if mask is not None:
                w = w + mask
            w = torch.softmax(w, dim=-1).to(torch.bfloat16)
            w = self.dropout(w)
            out = (w @ vh.to(torch.bfloat16)).to(torch.bfloat16)
            return out

        out_self  = attn_block(q_self,  k_self, v_self)
        assert not torch.isnan(out_self).any(), "NaN after self attn_block"
        out_cross = attn_block(q_cross, k_cross, v_cross)
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
        # Gemma applies an RMSNorm before every attention block; without it
        # hidden amplitudes (~50‑60) blow up the soft‑max. Re‑insert it here.
        self.pre_ln = nn.RMSNorm(hidden, eps=1e-5)
        self.cross_ln = nn.RMSNorm(hidden, eps=1e-5)
        # highways
        self.driver = GatedHighway(hidden)

        # High‑bandwidth feedback: full‑rank Linear + tanh‑bounded scalar gate.
        #   • weight zero‑init  → delta = 0 at t0
        #   • gate starts at 0  → tanh(0)=0 so path closed
        self.w_fb = nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)
        nn.init.zeros_(self.w_fb.weight)
        self.gate_fb = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))   # trainable scalar

    def forward(self, x_mod: torch.Tensor, x_back: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Standard block (4096‑d throughout): hybrid attention → driver → FFN → feedback.
        """
        # Gemma normalises inputs before attention – do the same
        x_q  = self.pre_ln(x_mod)
        x_kv = self.pre_ln(x_back)

        assert not torch.isnan(x_kv).any(), "NaN already in x_kv"
        assert not torch.isnan(x_q).any(),  "NaN already in x_q"
        attn_out = self.hybrid_attn(x_q, x_kv, mask)
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

def log_max(name):
    def _hook(_, __, output):
        # Gemma layers return either a tuple or BaseModelOutputWithPast; grab the tensor
        if isinstance(output, (tuple, list)):
            hidden = output[0]
        else:  # huggingface BaseModelOutput-like
            hidden = output.hidden_states if hasattr(output, "hidden_states") else output.last_hidden_state
        print(f"[{name}] max|x| = {hidden.abs().max().item():.2f}")
    return _hook

class GemmaModular(nn.Module):
    def __init__(self, predict: AutoModelForCausalLM, behave: AutoModelForCausalLM, layers: int = 8):
        
        super().__init__()
        #self.config = predict.config
        self.predict = predict.model
        # self.behave = behave.model
        assert predict.config.num_hidden_layers >= 8 and behave.config.num_hidden_layers >= 8, "Number of layers to slice larger than number in model"
        self.split = predict.config.num_hidden_layers - layers
        

        for p in self.predict.parameters():
            p.requires_grad = False
        self.predict.embed_tokens.requires_grad_(False)
        #self.predict.embed_positions.requires_grad_(False)
        # ✱A — one shared RoPE helper (lives on the same GPU as the first layer)
        # self.rotary_emb = GemmaRotaryEmbedding(self.config).to(
        #     next(self.backbone_layers.parameters()).device
        # )
        #self.embed = self.predict.model.embed_tokens; self.embed.requires_grad_(False)
        # -------------------------------------------------------------------
        # build module

        # build module blocks without spiking GPU RAM by deep-copying on CPU first
        self.behave = nn.ModuleList()
        # First ModuleBlock now starts at the model hidden size (4096)
        hidden_dim = behave.config.hidden_size   # Gemma‑7B = 4096
        prev_hidden = hidden_dim
        for bl in behave.model.layers[self.split:]:
            device = next(bl.parameters()).device
            hidden_curr = bl.self_attn.q_proj.out_features
            mod_block = ModuleBlock(bl, in_dim=prev_hidden)  # map prev → current
            # keep the module in BF16 so its Linear layers match BF16 activations
            mod_block.to(device, dtype=torch.bfloat16)
            self.behave.append(mod_block)
            prev_hidden = hidden_curr  # next block's "in_dim"
        # -------------------------------------------------------------------
        num_tokens = behave.model.embed_tokens.num_embeddings
        self.embed_delta = nn.Embedding(num_tokens, hidden_dim, dtype=torch.bfloat16)
        self.delta_gate = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        nn.init.zeros_(self.embed_delta.weight)
        # -------------------------------------------------------------------
        self.norm = behave.model.norm
        self.lm_head = behave.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        B, T = input_ids.shape
        h_back = self.embed(input_ids)
        # 1️⃣ give the backbone what it expects: an *additive* float mask
        if attention_mask is not None:
            # convert 1 → 0   and   0 → -65 000 (≈ -inf in fp16/bf16)
            attn_mask = (1.0 - attention_mask.to(h_back.dtype)) * torch.finfo(h_back.dtype).min
        else:
            attn_mask = None

        if self.pos is not None:
            h_back = h_back + self.pos(torch.arange(T, device=h_back.device))[None, :]
        assert not torch.isnan(h_back).any(), "NaN already in h_back from very beginning"
        assert not torch.isinf(h_back).any(), "Infinity already in h_back from very beginning"
        # frozen until split
        # build RoPE tuple for this sequence
        seq_len = h_back.size(1)
        position_ids = torch.arange(seq_len, device=h_back.device).unsqueeze(0).expand(B, -1)
        for idx, layer in enumerate(self.backbone_layers[:self.split]):
            cos, sin = self.rotary_emb(h_back, position_ids)
            layer.register_forward_hook(log_max(f"bb{idx}"))
            assert not torch.isinf(h_back).any(), "Infinity already in h_back {idx} layers in"
            assert not torch.isnan(h_back).any(), f"NaN already in h_back {idx} layers in"
            h_back = layer(
                h_back,
                position_embeddings=(cos, sin),                    # keep dtype consistent
                attention_mask=attn_mask,
                output_attentions=False,
            )[0]                                                   # layer already returns BF16
            
        assert not torch.isinf(h_back).any(), "Infinity already in h_back  layers in"
        # module initial state
        h_mod = h_back.clone()
        feedback = torch.zeros_like(h_back)
        # paired layers
        for idx, (predict_layer, behave_layer) in enumerate(
            zip(self.predict.layers[self.split:], self.behave.layers)
        ):
            print("‖before prediction‖", h_back.abs().max())
            cos, sin = self.rotary_emb(h_back, position_ids)
            h_back = predict_layer(
                h_back + feedback,
                position_embeddings=(cos, sin),             # ← NEW
                attention_mask=attn_mask,
                output_attentions=False
            )[0]
            print("‖after prediction‖", h_back.abs().max())
            # give the tuple to this block’s HybridAttention
            behave_layer.hybrid_attn.rotary = lambda x, pos=None, _cs=(cos, sin): _cs  # cos/sin already BF16
            assert not torch.isinf(h_back).any(), "Infinity already in h_back"
            h_mod, feedback = behave_layer(h_mod, h_back, attn_mask)
            # add tanh-gated delta embedding
            print(h_mod, self.embed_delta(input_ids), self.delta_gate)
            h_mod = h_mod + torch.tanh(self.delta_gate) * self.embed_delta(input_ids)
        logits = self.lm_head(self.norm(h_mod))

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
