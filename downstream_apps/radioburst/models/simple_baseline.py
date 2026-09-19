"""
A simple linear regression model to be used as a baseline for radio burst forecasting.
"""

import torch
import torch.nn as nn
from einops import rearrange


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


class TwoStageBurstModel(nn.Module):
    def __init__(self, input_dim: int):
        """
        Initializes the TwoStageBurstModel.

        Args:
            input_dim (int): The size of the input vector after channel and time dimensions are
                flattened. Since forward() concatenates spatial mean and std per channel/timestep,
                this should equal 2 * C * T.

        Note:
            This model expects 'ts' in the batch dict to already be in **signum-log** space
            (channel z-scores undone, log compression retained). Use
            destandardize_channels() to pre-process normalized SDO inputs before passing
            them here (e.g., via the preprocess_fn argument of RadioBurstLightningModule).

        The model no longer holds a copy of the median burst-spectrogram template. It used
        to (as a peak-normalized buffer it multiplied its amplitude prediction by), but
        once ``peak_amp`` is trained in ``SpectraTemplateNormalizer``'s standardized space
        (see ``spectra_transform.py``) that template relationship lives entirely inside
        the normalizer: this model's own generative assumption,
        ``spectra ≈ peak_amp · (template / template.max())``, collapses to
        ``standardized_peak_amp`` being *constant* across (t, f) once the template is
        divided out in log space, so reconstructing a full spectrogram is just
        broadcasting the scalar and letting ``SpectraTemplateNormalizer.inverse()`` add
        the template shape back — see that module's docstring for the derivation.
        """
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
        self.amplitude_regressor = nn.Linear(input_dim, 1)

    def forward(self, x: dict) -> dict:
        """
        Performs a forward pass through the model.

        Args:
            x (dict): Batch dict with 'ts' of shape (B, C, T, H, W) in signum-log space.

        B - Batch size
        C - Channels
        T - Time steps
        H - Height
        W - Width

        Returns:
            dict with:
                "burst_prob" (B, 1): sigmoid burst classification.
                "peak_amp" (B, 1): regressed amplitude, in ``SpectraTemplateNormalizer``'s
                    standardized space — the same space the ``"spectra"`` target from
                    ``RadioBurstDSDataset`` is in.
                "spectra" (B, 1, 1): ``peak_amp`` broadcast to a flat field. Pass it through
                    ``SpectraTemplateNormalizer.inverse()`` to reconstruct the full
                    ``(T, F)`` raw-flux spectrogram (the template shape is added back
                    there, not here).
        """
        x = x["ts"]

        # Collapse input stack spatially into per-channel/timestep mean and std. Signed mean is
        # kept (rather than abs()) since sign is physically meaningful for HMI polarity; std
        # captures spatial variability that a mean-only summary would discard.
        mean = x.mean(dim=[3, 4])
        std = x.std(dim=[3, 4])
        x = torch.cat([mean, std], dim=1)

        # Rearrange in preparation for linear layer
        x = rearrange(x, "b c t -> b (c t)")

        burst_prob = torch.sigmoid(self.classifier(x))
        peak_amp = self.amplitude_regressor(x)
        spectra = peak_amp.unsqueeze(-1)

        return {"burst_prob": burst_prob, "peak_amp": peak_amp, "spectra": spectra}