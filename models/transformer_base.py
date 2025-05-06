import torch
import torch.nn as nn

class CyberKindTransformer(nn.Module):
    def __init__(self, input_dim=25, embed_dim=64, num_heads=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=2
        )
        self.behavior_head = nn.Linear(embed_dim, 4)
        self.prediction_head = nn.Linear(embed_dim, 4)
        self.perception_head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x.unsqueeze(0)).squeeze(0)
        return {
            "behavior": self.behavior_head(x),
            "prediction": self.prediction_head(x),
            "perception": self.perception_head(x)
        }