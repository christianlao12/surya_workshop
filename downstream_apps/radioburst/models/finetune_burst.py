"""
Surya fine-tuning head for the radio-burst task: burst classification plus a full spectrogram.
"""

import math

import torch
from torch import nn

from workshop_infrastructure.models.finetune_models import HelioSpectformer1D


class HelioSpectformerBurst(HelioSpectformer1D):
    """HelioSpectformer1D with a burst logit and a rank-r bottleneck spectrogram decoder.

    The parent's ``head_unembed`` is sized to ``1 + spectra_rank`` outputs: column 0 is the
    burst logit, the remaining ``spectra_rank`` columns are coefficients that
    ``head_spectra_decoder`` maps onto a ``(T, F)`` spectrogram. The decoder's weight columns
    are ``spectra_rank`` learned basis spectrograms, so each prediction is a learned mix of
    them — a generalization of the linear baseline's single median template.

    A bottleneck rather than a flat ``Linear(embed_dim, T*F)`` readout keeps the head small
    (about 1.2M decoder parameters at r=32 for a 120x288 spectrogram, instead of ~44M).

    ``head_spectra_decoder`` follows the ``head_`` convention, so ``apply_peft_lora()`` keeps
    it trainable alongside the parent's head modules.

    Returns a dict:
        "burst_logit" (B, 1): raw, pre-sigmoid burst score. Losses should use
            ``binary_cross_entropy_with_logits``, which is also safe under bf16 autocast.
        "spectra" (B, T, F): predicted spectrogram, in whatever space the dataset's
            ``spectra_transform`` produced the targets in.
    """

    def __init__(self, *args, spectrum_shape: tuple[int, int], spectra_rank: int = 32, **kwargs):
        super().__init__(*args, num_outputs=1 + spectra_rank, **kwargs)
        self.spectrum_shape = tuple(spectrum_shape)
        self.head_spectra_decoder = nn.Linear(spectra_rank, math.prod(self.spectrum_shape))

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        out = super().forward(batch)  # (B, 1 + r)
        spectra = self.head_spectra_decoder(out[:, 1:])  # (B, T*F)
        return {
            "burst_logit": out[:, :1],
            "spectra": spectra.reshape(-1, *self.spectrum_shape),
        }
