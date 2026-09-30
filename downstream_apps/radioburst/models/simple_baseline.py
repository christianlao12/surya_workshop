"""
A linear baseline for Type II forecasting: per-channel image statistics -> two logits.

The fine-tuned Surya model has to beat this to show that the backbone adds anything
beyond "how bright / how magnetically active is the disk right now".
"""

import torch
import torch.nn as nn
from einops import rearrange

from downstream_apps.radioburst.labels import LABELS


def destandardize_channels(batch: dict, channel_order: list, scalers: dict) -> dict:
    """Return a new batch dict with 'ts' moved from normalized space to signum-log space.

    This undoes the per-channel z-score ONLY. The signum-log compression applied by the
    dataset is deliberately left in place, so the result is
    ``sign(x*s) * log1p(|x*s|)`` — not raw DN/Gauss. Values spanning many orders of
    magnitude make poor features for a single linear layer, so log space is what the
    baseline wants.

    If you need true physical units (plotting, a physical-space loss), use
    ``HelioNetCDFDataset.inverse_transform_data()`` instead, which undoes both stages.
    See the "THE THREE SPACES" block in ``workshop_infrastructure/datasets/helio.py``.

    Args:
        batch: Batch dict containing at minimum a 'ts' key with shape (B, C, T, H, W).
        channel_order: Channel names in the same order as the C dimension of 'ts'.
        scalers: Dict mapping channel name -> scaler with an inverse_transform method.

    Returns:
        A new batch dict with 'ts' replaced by the de-standardized (signum-log) tensor.
    """
    x = batch["ts"].clone()
    with torch.no_grad():
        for i, channel in enumerate(channel_order):
            x[:, i, ...] = scalers[channel].inverse_transform(x[:, i, ...])
    return {**batch, "ts": x}


class LinearTypeIIModel(nn.Module):
    """One linear layer on the spatial mean and standard deviation of every channel.

    Expects ``batch["ts"]`` in **signum-log** space (pass ``destandardize_channels`` as
    the Lightning module's ``preprocess_fn``).

    Args:
        input_dim: ``2 * C * T`` — a mean and a std per channel and input timestep.

    Returns (from ``forward``):
        Logits of shape ``(B, 2)``, in ``LABELS`` order — the same contract as
        ``HelioSpectformer1D`` with ``num_outputs=2``, so both models share one loss.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, len(LABELS))

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["ts"]  # (B, C, T, H, W)
        # The signed mean is kept (sign matters for HMI polarity); the std captures the
        # spatial structure a mean alone would discard.
        features = torch.cat([x.mean(dim=(3, 4)), x.std(dim=(3, 4))], dim=1)  # (B, 2C, T)
        return self.linear(rearrange(features, "b c t -> b (c t)"))
