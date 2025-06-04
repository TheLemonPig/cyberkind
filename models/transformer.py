import torch
import torch.nn as nn

# Shared gated transformer layer that processes both behavior and prediction with gradient gating
class GatedTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.behavior_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.prediction_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)

    def forward(self, beh_in, pred_in):
        beh_out = self.behavior_layer(beh_in)
        pred_out = self.prediction_layer(pred_in)

        # Cross communication with detached signals
        beh_out = beh_out + pred_out.detach()
        pred_out = pred_out + beh_out.detach()

        return beh_out, pred_out

# Perception module
class PerceptionModule(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.perception_head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        perception = self.perception_head(x)
        return x, perception

# Combined stack for behavior and prediction modules with gated layers
class BehaviorPredictionStack(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedTransformerLayer(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        beh = x.unsqueeze(0)  # [batch, seq=1, dim]
        pred = x.unsqueeze(0)

        for layer in self.layers:
            beh, pred = layer(beh, pred)

        return beh.squeeze(0), pred.squeeze(0)

# Top-level model
class CyberKindModel(nn.Module):
    def __init__(self, input_dim=25, num_actions=4, embed_dim=64, num_heads=2, num_layers=2):
        super().__init__()
        self.perception = PerceptionModule(input_dim, embed_dim)
        self.beh_pred_stack = BehaviorPredictionStack(num_layers, embed_dim, num_heads)
        self.behavior_head = nn.Linear(embed_dim, num_actions)
        self.prediction_head = nn.Linear(embed_dim, input_dim)

    def forward(self, x):
        embedding, perception_out = self.perception(x)
        behavior_repr, prediction_repr = self.beh_pred_stack(embedding)
        behavior_out = self.behavior_head(behavior_repr)
        prediction_out = self.prediction_head(prediction_repr)

        return {
            "perception": perception_out,
            "behavior": behavior_out,
            "prediction": prediction_out
        }