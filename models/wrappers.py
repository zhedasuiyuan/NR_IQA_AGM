"""
Composite model wrappers used for GradCAM visualisation and standalone inference.
Author: Ankit Yadav
"""
import warnings
import torch.nn as nn

from .multi_layer_fusion import extract_token_features, native_pool


class SIGLIPWithMLP(nn.Module):
    """Wraps a vision backbone (SigLIP / DINOv2 / ResNet) together with an MLP
    head so that GradCAM can be applied end-to-end.

    Args:
        base_model: the vision encoder (e.g. SigLIP2 ``AutoModel``).
        mlp_head: the quality-prediction MLP (e.g. ``MLP3_Gated``).
        device: torch device.
        layer: layer index used by perception-style encoders.
        resnet: set True when the backbone is a ResNet.
        fusion: optional ``MultiLayerFusion`` module. When set, features come
            from tapping multiple layers, fusing them at the token level, and
            running the backbone's native pooler -- instead of the single-layer
            ``get_image_features`` path.
    """

    def __init__(self, base_model, mlp_head, device, layer=18, resnet=False, fusion=None):
        super().__init__()
        self.siglip   = base_model
        self.mlp_head = mlp_head
        self.device   = device
        self.layer    = layer
        self.resnet   = resnet
        self.fusion   = fusion

    def forward(self, inputs):
        if self.fusion is not None:
            feats, trunk = extract_token_features(self.siglip, inputs, self.fusion.layer_indices)
            fused = self.fusion(feats, trunk)
            # ALF returns the pooled [B, D] vector directly; others return tokens.
            features = (fused if getattr(self.fusion, "returns_pooled", False)
                        else native_pool(self.siglip, fused))
            scores = self.mlp_head(features)
            return scores.squeeze(1)

        if self.resnet:
            warnings.warn(
                "ResNet152 backbone detected — using pooler_output. "
                "If this is not intended, check the backbone type."
            )
            features = self.siglip(**inputs).pooler_output
            features = features.squeeze(-1).squeeze(-1)
        elif not (getattr(getattr(self.siglip, "config", None), "model_type", "")
                  or "").lower().startswith("siglip"):
            # CLIP's get_image_features returns the projection dim (not hidden) and
            # DINOv2 has none -> pool the last hidden state to hidden dim, matching
            # train.py's no-fusion path. SigLIP keeps get_image_features (below).
            _, trunk = extract_token_features(self.siglip, inputs, [1])
            features = native_pool(self.siglip, trunk)
        else:
            try:
                features = self.siglip.get_image_features(inputs)
            except Exception:
                warnings.warn(
                    "SigLIP get_image_features failed — falling back to "
                    "DINOv2-style average-pooled last_hidden_state."
                )
                try:
                    features = self.siglip(inputs).last_hidden_state.mean(dim=1)
                except Exception:
                    warnings.warn(
                        "DINOv2 fallback failed — falling back to "
                        "Perception-style encode_image_layers. "
                        "If this is not intended, check the backbone type."
                    )
                    features = self.siglip.encode_image_layers(
                        inputs, layer_idx=self.layer
                    )

        scores = self.mlp_head(features)  # (B, 1)
        return scores.squeeze(1)          # (B,)
