"""
Normalization of the radio-burst spectrogram target, and its inverse.

The fine-tuning head regresses a full ``(T, F)`` spectrogram. Raw flux is strictly
positive and spans about six orders of magnitude (~1e-15 to 1e-9), so an MSE on raw
values is vanishingly small and the brightest bins dominate. The recipe here is the one
the flare template applies to X-ray flux: ``log10``, shift so the minimum is 0, divide by
twice the standard deviation.

**Why this lives in a module rather than in the training notebook.** Predictions come back
in normalized space, so turning one into flux needs the exact ``minimum`` and ``scale``
that were used to build the targets. Defining the transform in a notebook cell leaves
those constants alive only for as long as the kernel is, with no inverse anywhere — which
is why ``SpectraLog10Normalizer`` owns both directions and can be fitted, saved and
reloaded. Training (``2_finetune_template_1D.ipynb``) and inference
(``4_inference_template.ipynb``) use this one definition.

The statistics are global over every spectrogram in the catalog, and are deliberately
fitted before the train/val split, so both sets are scaled identically — mirroring how
``label_transform`` is applied to ``FlareDSDataset``'s ``intensity`` column in
``downstream_apps/template``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# The spectra CSVs carry a leading timestamp column; everything after it is a frequency bin.
SPECTRA_TIME_COLUMN = "time"


def read_spectra_file(folder_path, relative_path) -> np.ndarray:
    """Load one spectra CSV as a ``(n_timesteps, n_bins)`` float32 array.

    Args:
        folder_path: Folder the catalog's relative paths are resolved against.
        relative_path: Path of the spectra CSV inside that folder.

    Returns:
        The frequency-bin columns, with the leading ``time`` column dropped.
    """
    return (
        pd.read_csv(Path(folder_path) / relative_path)
        .drop(columns=SPECTRA_TIME_COLUMN)
        .to_numpy(dtype=np.float32)
    )


def load_catalog_spectra(folder_path, index_file: str, spectra_column: str) -> pd.Series:
    """Load every spectrogram referenced by a radio-burst catalog.

    Args:
        folder_path: Folder holding the catalog and the spectra files.
        index_file: Catalog CSV filename inside ``folder_path``.
        spectra_column: Catalog column naming each window's spectra file.

    Returns:
        One ``(n_timesteps, n_bins)`` array per catalog row, in catalog order.
    """
    folder_path = Path(folder_path)
    catalog = pd.read_csv(folder_path / index_file)
    return catalog[spectra_column].apply(lambda p: read_spectra_file(folder_path, p))


@dataclass(frozen=True)
class SpectraLog10Normalizer:
    """log10 + shift + scale, with the constants kept so the transform can be undone.

    Attributes:
        minimum: The global ``nanmin`` of log10(flux) over the fitted population.
        scale: Twice the global ``nanstd`` of the shifted log10 values.

    Missing cells stay ``NaN`` in both directions: about 0.1% of catalog cells are gaps,
    and filling them would invent data. The statistics are NaN-aware and the loss skips
    them (see ``_masked_values`` in ``metrics/radioburst_metrics.py``).
    """

    minimum: float
    scale: float

    @classmethod
    def fit(cls, raw: pd.Series) -> "SpectraLog10Normalizer":
        """Fit the constants on a Series of raw-flux spectrogram arrays."""
        # A lambda, not np.log10 itself: Series.apply(ufunc) dispatches the ufunc over the
        # whole Series, which fails on a Series whose elements are arrays.
        log_spectra = raw.apply(lambda spectrum: np.log10(spectrum))
        stacked = np.stack(log_spectra.to_list())
        minimum = float(np.nanmin(stacked))
        scale = float(2 * np.nanstd(stacked - minimum))
        if not np.isfinite(minimum) or not np.isfinite(scale) or scale == 0:
            raise ValueError(
                f"Degenerate spectra statistics (minimum={minimum}, scale={scale}). "
                "Check the catalog for non-positive or all-NaN spectrograms."
            )
        return cls(minimum=minimum, scale=scale)

    @classmethod
    def fit_from_catalog(cls, cfg) -> "SpectraLog10Normalizer":
        """Fit on every spectrogram in the catalog named by ``cfg.data``.

        This is the same population ``RadioBurstDSDataset`` normalizes against at training
        time — the whole catalog, before the train/val split — so a fit here reproduces
        the constants a training run used, as long as the catalog itself has not changed.
        """
        return cls.fit(
            load_catalog_spectra(
                cfg.data.ds_radioburst_folder_path,
                cfg.data.ds_radioburst_index_file,
                cfg.data.ds_spectra_column,
            )
        )

    def __call__(self, raw: pd.Series) -> pd.Series:
        """Normalize a Series of raw-flux arrays. Drop-in for ``spectra_transform=``."""
        return raw.apply(
            lambda spectrum: ((np.log10(spectrum) - self.minimum) / self.scale).astype(np.float32)
        )

    def inverse(self, normalized) -> np.ndarray:
        """Undo the transform: normalized values back to raw flux.

        Args:
            normalized: Array (or tensor-like) of normalized values, any shape.

        Returns:
            The same shape in physical flux units, as float64 — the exponentiation
            recovers values around 1e-15, too small to keep precision in float32.
        """
        normalized = np.asarray(normalized, dtype=np.float64)
        return np.power(10.0, normalized * self.scale + self.minimum)

    def to_json(self, path) -> Path:
        """Write the constants beside a checkpoint, pinning what that run was trained on."""
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    @classmethod
    def from_json(cls, path) -> "SpectraLog10Normalizer":
        """Load constants written by :meth:`to_json`."""
        return cls(**json.loads(Path(path).read_text()))
