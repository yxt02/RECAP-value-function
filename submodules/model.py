from contextlib import nullcontext
import torch
from torch import nn
from transformers import SiglipVisionModel


class ValueModel(nn.Module):
    """SigLIP mean-patch features + distributional value head.

    Outputs a 201-bin categorical distribution over [-1, 0] and its expectation.
    Gemma was not used in the previous forward pass. Avoid allocating it.
    Feature ordering is the explicit camera ordering saved in each checkpoint.
    """
    def __init__(self, siglip_path, cameras=('cam2',), freeze_vlm=True,
                 precision='bf16', projection_dim=640, load_encoder=True, num_bins=201):
        super().__init__()
        self.cameras = tuple(cameras)
        self.freeze_vlm = freeze_vlm
        self.siglip = None
        self.num_bins = num_bins
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
        # Distributional head: outputs logits for num_bins bins
        self.value_head = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_bins)
        )
        # Register bin centers as buffer (not trainable)
        # Bins span [-1, 0] uniformly
        bin_edges = torch.linspace(-1, 0, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.register_buffer('bin_centers', bin_centers)
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

    def predict_distribution(self, features):
        """Return log-probabilities over bins. Shape: [B, num_bins]."""
        projected = self.projection(features.to(self.projection.weight.dtype))
        logits = self.value_head(projected).float()
        return torch.log_softmax(logits, dim=-1)

    def predict_features(self, features):
        """Return expected value from the distribution. Shape: [B, 1]."""
        log_probs = self.predict_distribution(features)
        probs = log_probs.exp()
        # Expectation: sum(prob * bin_center)
        expectation = (probs * self.bin_centers).sum(dim=-1, keepdim=True)
        return expectation

    def forward(self, images=None, features=None, return_distribution=False):
        """Forward pass.

        Args:
            images: Dict of camera images (for online encoding).
            features: Pre-extracted features (for cached training).
            return_distribution: If True, return (expectation, log_probs).

        Returns:
            expectation: [B, 1] expected value in [-1, 0].
            If return_distribution=True, also returns log_probs: [B, num_bins].
        """
        if features is None:
            features = self.encode(images)
            if self.freeze_vlm:
                # Match the float16 frozen-feature cache during online inference.
                features = features.to(torch.float16)
        log_probs = self.predict_distribution(features)
        expectation = (log_probs.exp() * self.bin_centers).sum(dim=-1, keepdim=True)
        if return_distribution:
            return expectation, log_probs
        return expectation
