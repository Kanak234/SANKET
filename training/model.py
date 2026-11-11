"""model.py — SANKET's own sign-recognition network (trained from scratch, random init).

WHY a transformer: a sign is a short movement over time. Self-attention lets every frame look at
     every other frame, so the model can relate "hand starts at chin" to "hand ends at chest".
     Kept deliberately small (~1.9 M parameters) so it runs in ~10 ms on a laptop CPU and later
     on a phone, and so 3,100 clips are enough to train it without huge overfitting.
WHAT: input (B, 64, 239) → linear embedding 192 → + learned position embedding → [CLS] token →
      4 pre-norm transformer layers (6 heads) → LayerNorm → linear → (B, 263) class scores.
      No pretrained weights anywhere (Kanak's model rules: no fine-tuning).
NEXT: train.py trains it; export_onnx.py saves it as models/recognizer.onnx for the app.
"""

from __future__ import annotations

import torch
from torch import nn

from sanket.core.normalize import FEATURES, SEQ_LEN


class SignTransformer(nn.Module):
    def __init__(self, n_classes: int = 263, d_model: int = 192, layers: int = 4, heads: int = 6,
                 ff: int = 384, dropout: float = 0.2) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(FEATURES, d_model), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))  # learned "summary" token
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN + 1, d_model))  # learned positions (frame order)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self._init()

    def _init(self) -> None:
        """Small random starting values (standard practice for transformers)."""
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 64, 239) features from sanket.core.normalize.model_input → (B, n_classes) logits."""
        h = self.embed(x)
        # Repeat the [CLS] token for every clip in the batch. Broadcasting against a zero slice of
        # `h` (instead of expand(len(h))) keeps the batch size dynamic in the exported ONNX file —
        # the parity test caught this: expand(len(h)) froze the batch size at export time.
        cls = self.cls + torch.zeros_like(h[:, :1, :])
        h = torch.cat([cls, h], dim=1) + self.pos
        h = self.encoder(h)
        return self.head(self.norm(h[:, 0]))  # classify from the [CLS] token


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
