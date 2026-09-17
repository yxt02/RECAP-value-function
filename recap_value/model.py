from contextlib import nullcontext
import torch
from torch import nn
from transformers import SiglipVisionModel


class ValueModel(nn.Module):
    """SigLIP mean-patch features + scalar regressor; not a Gemma/RECAP critic.

    Gemma was not used in the previous forward pass. Avoid allocating it.
    Feature ordering is the explicit camera ordering saved in each checkpoint.
    """
    def __init__(self, siglip_path, cameras=('cam2',), freeze_vlm=True,
                 precision='bf16', projection_dim=640, load_encoder=True):
        super().__init__()
        self.cameras = tuple(cameras)
        self.freeze_vlm = freeze_vlm
        self.siglip = None
        from transformers import SiglipVisionConfig
        config = SiglipVisionConfig.from_pretrained(siglip_path, local_files_only=True)
        self.feature_dim = config.hidden_size * len(self.cameras)
        if load_encoder:
            self.siglip = SiglipVisionModel.from_pretrained(
                siglip_path, local_files_only=True,
                dtype=torch.bfloat16 if precision == 'bf16' else torch.float32,
                attn_implementation='sdpa')
            self.siglip.requires_grad_(not freeze_vlm)
        self.projection = nn.Linear(self.feature_dim, projection_dim)
        self.value_head = nn.Sequential(nn.Linear(projection_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.siglip is not None and self.freeze_vlm:
            self.siglip.eval()
        return self

    def encode(self, images):
        if self.siglip is None:
            raise RuntimeError('Encoder not loaded: use cached features or load_encoder=True')
        with torch.no_grad() if self.freeze_vlm else nullcontext():
            # No CLS token in SigLIP. Preserve every configured camera.
            return torch.cat([self.siglip(images[f'observation.images.{cam}']).last_hidden_state.mean(1)
                              for cam in self.cameras], dim=-1)

    def predict_features(self, features):
        # CPU/fp32 mode also accepts the float16 disk cache.
        projected = self.projection(features.to(self.projection.weight.dtype))
        return (torch.tanh(self.value_head(projected).float()) - 1.) * .5

    def forward(self, images=None, features=None):
        if features is None:
            features = self.encode(images)
            if self.freeze_vlm:
                # Match the float16 frozen-feature cache during online inference.
                features = features.to(torch.float16)
        return self.predict_features(features)
