"""K=3 transformer for asteroid pole prediction.

Architecture:
- Token projection: F -> d_model + LayerNorm
- Window encoder: TransformerEncoder over tokens per window (GELU)
- Attention pooling: attention-based pooling per window
- Mean pooling across windows (for W > 1)
- K=3 slot heads: separate MLPs (3 poles)
- Outputs: poles [3, 3] (unit vectors) and optional quality_logits [3]
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WindowTransformer(nn.Module):
    """Transformer encoder for tokens within a single window."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class AttentionPooling(nn.Module):
    """Attention-based pooling over token sequence to [d_model]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.pool_query, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = self.pool_query.expand(x.size(0), -1, -1)
        scores = torch.matmul(query, x.transpose(1, 2)) / (self.pool_query.size(-1) ** 0.5)
        scores = scores + (1 - mask).unsqueeze(1) * (-1e9)
        attn = F.softmax(scores, dim=-1)
        pooled = torch.matmul(attn, x)
        return pooled.squeeze(1)


class K3SlotHead(nn.Module):
    """Independent MLP head for one slot (pole + optional quality)."""

    def __init__(self, d_model: int, include_quality: bool = False):
        super().__init__()
        self.include_quality = include_quality

        self.pole_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

        if include_quality:
            self.quality_mlp = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pole_raw = self.pole_mlp(z)
        poles = F.normalize(pole_raw, p=2, dim=-1)

        quality = None
        if self.include_quality:
            quality = self.quality_mlp(z)

        return poles, quality


class GeoHierK3Transformer(nn.Module):
    """Production K=3 transformer model for asteroid pole prediction.

    Architecture: token projection → Transformer encoder → attention pooling
    → mean-pool across windows → 3 independent MLP slot heads → 3 unit-vector
    pole candidates.

    ~994K parameters (d_model=128, n_heads=4, 4 layers, GELU).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        n_feature_input: int = 13,
        include_quality_head: bool = False,
        dropout: float = 0.1,
        **kwargs,  # Accept and ignore legacy params (n_layers_window, n_layers_cross)
    ):
        super().__init__()
        self.d_model = d_model
        self.include_quality_head = include_quality_head

        # Token projection with normalization
        self.token_proj = nn.Linear(n_feature_input, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Single encoder path
        self.encoder = WindowTransformer(d_model, n_heads, n_layers, dropout)
        self.pool = AttentionPooling(d_model)

        # K=3 slot heads
        self.slot_head_0 = K3SlotHead(d_model, include_quality=include_quality_head)
        self.slot_head_1 = K3SlotHead(d_model, include_quality=include_quality_head)
        self.slot_head_2 = K3SlotHead(d_model, include_quality=include_quality_head)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens: [B, W, T, F] token sequences
            mask: [B, W, T] padding mask (1 = valid)

        Returns:
            (poles [B, 3, 3], quality_logits [B, 3] or None)
        """
        B, W, T, F = tokens.shape

        # Flatten windows into batch
        tokens_flat = tokens.view(B * W, T, F)
        mask_flat = mask.view(B * W, T)

        # Project and normalize
        projected = self.token_proj(tokens_flat)
        projected = self.input_norm(projected)

        # Encode
        key_pad_mask = (1 - mask_flat).bool()
        encoded = self.encoder(projected, key_pad_mask)

        # Pool tokens -> single embedding per window
        pooled = self.pool(encoded, mask_flat)

        # Reshape and mean-pool across windows
        pooled = pooled.view(B, W, self.d_model)
        z = pooled.mean(dim=1)  # [B, d_model]

        # K=3 slot heads
        poles_0, quality_0 = self.slot_head_0(z)
        poles_1, quality_1 = self.slot_head_1(z)
        poles_2, quality_2 = self.slot_head_2(z)

        poles = torch.stack([poles_0, poles_1, poles_2], dim=1)

        quality_logits = None
        if self.include_quality_head:
            quality_logits = torch.cat([quality_0, quality_1, quality_2], dim=1)

        return poles, quality_logits


def load_checkpoint(checkpoint_path: str) -> Tuple[GeoHierK3Transformer, dict]:
    """Load model and config from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})

    state_dict = ckpt['model_state_dict']
    has_quality = any('quality_mlp' in k for k in state_dict.keys())
    include_quality_head = config.get('include_quality_head', has_quality)

    model = GeoHierK3Transformer(
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 4),
        n_feature_input=config.get('n_feature_input', 13),
        include_quality_head=include_quality_head,
    )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        if 'Unexpected key' in str(e) or 'Missing key' in str(e):
            model_state = model.state_dict()
            filtered_state = {k: v for k, v in state_dict.items() if k in model_state}
            model.load_state_dict(filtered_state, strict=False)
            logger.warning(f"Loaded checkpoint with partial state dict (filtered {len(state_dict) - len(filtered_state)} keys)")
        else:
            raise

    return model, config


# Alias for imports that reference the PolePredictor name
PolePredictor = GeoHierK3Transformer
