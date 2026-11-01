"""Condition adapters whose internal width is independent of token width."""
from torch import nn
from timm.models.vision_transformer import RmsNorm


class ResidualMLPAdapter(nn.Module):
    """Pre-RMSNorm residual FFN; the shortcut retains the original token."""

    def __init__(self, in_features, out_features):
        super().__init__()
        if in_features != out_features:
            raise ValueError("ResidualMLPAdapter requires matching input/output widths")
        hidden_features = 2 * out_features
        self.norm = RmsNorm(in_features, eps=1e-6)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, tokens):
        return tokens + self.fc2(self.act(self.fc1(self.norm(tokens))))
