import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

# -----------------------------
# 1.  Base (frozen) backbone
# -----------------------------

BACKBONE_ID = "google/gemma-7b"

# 8‑bit load keeps the run under a single A100‑40 GB
def load_frozen_backbone(model_id: str = BACKBONE_ID):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_8bit=True,
        output_hidden_states=True,
    )

    for p in model.parameters():
        p.requires_grad = False
    return model

# -----------------------------
# 2.  Module building blocks
# -----------------------------

class ModuleCrossLayer(nn.Module):
    """One transformer layer that
       • receives the previous module hidden state y_prev (B, T, D_mod)
       • receives a *detached* backbone hidden state x_det (B, T, D_back)
       and returns the next hidden state.
       It implements:
         y_hat = LN(    y_prev
                       + CrossAttn(Q=y_prev, K=x_det, V=x_det)
                       + HighwayProj(pool(x_det))
                 )
       followed by a standard FFN.
    """

    def __init__(self, d_mod: int, d_back: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_mod, d_mod)
        self.k_proj = nn.Linear(d_back, d_mod)
        self.v_proj = nn.Linear(d_back, d_mod)
        self.attn = nn.MultiheadAttention(d_mod, n_heads, dropout=dropout, batch_first=True)

        self.highway_proj = nn.Linear(d_back, d_mod)
        self.norm1 = nn.LayerNorm(d_mod)

        # FFN
        self.ff = nn.Sequential(
            nn.Linear(d_mod, 4 * d_mod),
            nn.GELU(),
            nn.Linear(4 * d_mod, d_mod),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_mod)

    def forward(self, y_prev: torch.Tensor, x_det: torch.Tensor):
        # y_prev, x_det -> (B, T, *)
        q = self.q_proj(y_prev)
        k = self.k_proj(x_det)
        v = self.v_proj(x_det)
        attn_out, _ = self.attn(q, k, v, need_weights=False)

        # Highway: pool over sequence length (mean) then project & broadcast
        highway = self.highway_proj(x_det.mean(dim=1, keepdim=True))
        highway = highway.expand_as(y_prev)

        y = self.norm1(y_prev + attn_out + highway)
        y = self.norm2(y + self.ff(y))
        return y


class ModuleStack(nn.Module):
    """Stack of ModuleCrossLayers."""

    def __init__(self, num_layers: int, d_mod: int, d_back: int, n_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            ModuleCrossLayer(d_mod, d_back, n_heads=n_heads) for _ in range(num_layers)
        ])

        # If d_mod != d_back we need a bridge for the first layer
        self.bridge = (
            nn.Linear(d_back, d_mod) if d_back != d_mod else nn.Identity()
        )

    def forward(self, backbone_hiddens):
        """backbone_hiddens: list[Tensor] length >= num_layers.
           Returns final module hidden (B, T, d_mod)
        """
        y = self.bridge(backbone_hiddens[0])  # first driver layer
        for i, layer in enumerate(self.layers):
            y = layer(y, backbone_hiddens[i])
        return y


# -----------------------------
# 3.  Full wrapper model
# -----------------------------

class GemmaWithModule(nn.Module):
    def __init__(self, backbone: nn.Module, module_depth: int = 8, module_dim: int = 1024):
        super().__init__()
        self.backbone = backbone
        d_back = backbone.config.hidden_size  # 3072 for Gemma‑7B
        self.module = ModuleStack(module_depth, module_dim, d_back)
        self.lm_head = nn.Linear(module_dim, backbone.config.vocab_size, bias=False)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.forward_generate(*args, **kwargs)

    def forward(self, input_ids, attention_mask=None):
        # Pass through Gemma (frozen) – keep hidden states
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states  # tuple length (L+1)
        # Take the last `module_depth` layers (or pad/repeat if shorter)
        needed = len(self.module.layers)
        if len(hidden_states) < needed + 1:
            raise ValueError("Backbone has fewer layers than module depth")
        x_hiddens = [h.detach() for h in hidden_states[-needed:]]

        y = self.module(x_hiddens)  # (B, T, d_mod)
        logits = self.lm_head(y)
        return {"logits": logits}


# -----------------------------
# 4.  Minimal training harness
# -----------------------------

def build_model(device="cuda"):
    backbone = load_frozen_backbone()
    model = GemmaWithModule(backbone, module_depth=8, module_dim=1024)
    model.to(device)
    return model


def main():
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_ID)
    model = build_model()

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


if __name__ == "__main__":
    main()
