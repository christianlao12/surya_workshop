"""
Surya fine-tuning head for the radio-burst task: burst classification plus a full spectrogram.
"""

import math

import torch
from torch import nn

from workshop_infrastructure.models.finetune_models import HelioSpectformer1D


class HelioSpectformerBurst(HelioSpectformer1D):
    """HelioSpectformer1D with two readouts of the pooled embedding: a burst logit and a
    rank-r spectrogram.

    Both heads read the same ``(B, embed_dim)`` vector from the parent's
    ``forward_features()``. The parent's ``head_unembed`` stays a plain single-output layer
    for the burst logit; ``head_spectra`` is the spectrogram decoder. Its second layer's
    weight columns are ``spectra_rank`` learned basis spectrograms, so each prediction is a
    learned mix of them — a generalization of the linear baseline's single median template.

    A rank-r bottleneck rather than a flat ``Linear(embed_dim, T*F)`` readout keeps the head
    small: about 1.2M parameters at r=32 for a 120x288 spectrogram, instead of ~44M.

    ``head_spectra`` follows the ``head_`` convention, so ``apply_peft_lora()`` keeps it
    trainable alongside the parent's head modules.

    Returns a dict:
        "burst_logit" (B, 1): raw, pre-sigmoid burst score. Losses should use
            ``binary_cross_entropy_with_logits``, which is also safe under bf16 autocast.
        "spectra" (B, T, F): predicted spectrogram, in whatever space the dataset's
            ``spectra_transform`` produced the targets in.
    """

    def __init__(self, *args, spectrum_shape: tuple[int, int], spectra_rank: int = 32, **kwargs):
        super().__init__(*args, num_outputs=1, **kwargs)
        self.spectrum_shape = tuple(spectrum_shape)
        # No activation between the two layers on purpose: this is a rank-r factorization
        # of one embed_dim -> T*F projection, not an MLP.
        self.head_spectra = nn.Sequential(
            nn.Linear(self.embed_dim, spectra_rank),
            nn.Linear(spectra_rank, math.prod(self.spectrum_shape)),
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        features = self.forward_features(batch)  # (B, embed_dim)
        return {
            "burst_logit": self.head_unembed(features),
            "spectra": self.head_spectra(features).reshape(-1, *self.spectrum_shape),
        }
