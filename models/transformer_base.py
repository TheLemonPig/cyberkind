import torch
import torch.nn as nn

# Perception Module
class PerceptionModule(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.perception_head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        perception = self.perception_head(x)
        return x, perception  # x is embedding, perception is output

# Behavior Module
class BehaviorModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.behavior_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=1
        )
        self.behavior_head = nn.Linear(embed_dim, 4)

    def forward(self, x):
        # x shape: (batch, embed_dim)
        # Transformer expects (seq_len, batch, embed_dim)
        # We'll treat batch as seq_len=1 if needed
        behavior_x = self.behavior_transformer(x.unsqueeze(0)).squeeze(0)
        behavior = self.behavior_head(behavior_x)
        return behavior

# Prediction Module
class PredictionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.prediction_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=1
        )
        self.prediction_head = nn.Linear(embed_dim, 4)

    def forward(self, x):
        prediction_x = self.prediction_transformer(x.unsqueeze(0)).squeeze(0)
        prediction = self.prediction_head(prediction_x)
        return prediction

# CyberKindModel combines the three modules
class CyberKindModel(nn.Module):
    def __init__(self, input_dim=25, embed_dim=64, num_heads=2):
        super().__init__()
        self.perception = PerceptionModule(input_dim, embed_dim)
        self.behavior = BehaviorModule(embed_dim, num_heads)
        self.prediction = PredictionModule(embed_dim, num_heads)

    def forward(self, x):
        embedding, perception_out = self.perception(x)
        behavior_out = self.behavior(embedding)
        prediction_out = self.prediction(embedding)
        return {
            "perception": perception_out,
            "behavior": behavior_out,
            "prediction": prediction_out
        }