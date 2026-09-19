"""
Normalization of the radio-burst spectrogram target, and its inverse.

The fine-tuning head regresses a full ``(T, F)`` spectrogram, and the linear baseline
(``TwoStageBurstModel``) regresses a scalar that reconstructs one. Raw flux is strictly
positive and spans about six orders of magnitude (~1e-15 to 1e-9), so an MSE on raw
values is vanishingly small and the brightest bins dominate.

``SpectraTemplateNormalizer`` divides out each event's typical shape before scaling: a
precomputed median burst spectrogram (``ds_spectra_template_file``) — the same template
``TwoStageBurstModel`` already uses to turn a scalar amplitude into a full spectrogram.
Working in log space, "dividing out the template" becomes subtracting
``log10(template)``, which stays safe wherever the template is small (a literal division
would not). What is left over is a residual that a single global median/MAD standardizes
into a comparable, roughly-unit-scale target. Median/MAD rather than mean/std: most cells
are quiet background and a few are genuine burst excursions, and a plain std lets those
excursions inflate the very scale meant to give them contrast.

**Why this lives in a module rather than in the training notebook.** Predictions come
back in normalized space, so turning one into flux needs the exact ``log_template``,
``center`` and ``scale`` that were used to build the targets. Defining the transform in a
notebook cell leaves those constants alive only for as long as the kernel is, with no
inverse anywhere — which is why ``SpectraTemplateNormalizer`` owns both directions and
can be fitted, saved and reloaded. Training (``2_finetune_template_1D.ipynb``) and
inference (``4_inference_template.ipynb``) use this one definition.

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

# The standard MAD -> sigma-equivalent constant, so `scale` sits on the same footing as a std.
_MAD_TO_SIGMA = 1.4826


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


def load_median_template(folder_path, template_file: str) -> np.ndarray:
    """Load the precomputed median burst-spectrogram template.

    The template file mirrors an individual burst spectra file's shape — a leading
    non-value column (a timestamp, or for the template a nominal step label) followed by
    one column per frequency bin — but isn't required to use the same column names, so
    the leading column is dropped by position rather than by name. Shared by
    ``RadioBurstDSDataset`` and ``SpectraTemplateNormalizer.fit_from_catalog``, so there
    is one definition of "the template" rather than copies that could drift apart.

    Args:
        folder_path: Folder the catalog's relative paths are resolved against.
        template_file: Filename of the template CSV inside ``folder_path``.

    Returns:
        (T, F) array of the frequency-bin columns.
    """
    return (
        pd.read_csv(Path(folder_path) / template_file)
        .iloc[:, 1:]
        .to_numpy(dtype=np.float32)
    )


@dataclass(frozen=True)
class SpectraTemplateNormalizer:
    """log10, subtract the median-template shape, then a global median/MAD scale.

    Attributes:
        log_template: ``log10`` of the median burst-spectrogram template, ``(T, F)``,
            computed once at fit time so ``__call__``/``inverse`` don't recompute it.
        center: The global ``nanmedian`` of ``log10(raw) - log_template`` over the fitted
            population.
        scale: The global, robust spread of that same residual:
            ``1.4826 * nanmedian(|residual - center|)`` — MAD, scaled to match a std under
            a normal distribution.

    Missing cells stay ``NaN`` in both directions: about 0.1% of catalog cells are gaps,
    and filling them would invent data. The statistics are NaN-aware and the loss skips
    them (see ``_masked_values`` in ``metrics/radioburst_metrics.py``).
    """

    log_template: np.ndarray
    center: float
    scale: float

    @classmethod
    def fit(cls, raw: pd.Series, template: np.ndarray) -> "SpectraTemplateNormalizer":
        """Fit the constants on a Series of raw-flux spectrogram arrays and their template.

        Args:
            raw: Catalog spectrograms, each ``(T, F)``, matching ``template``'s shape.
            template: The median burst-spectrogram template, ``(T, F)``. Must be finite
                and strictly positive everywhere — see the raised error below for why.
        """
        template = np.asarray(template, dtype=np.float32)
        if not (np.all(np.isfinite(template)) and np.all(template > 0)):
            raise ValueError(
                "The median template must be finite and strictly positive everywhere: a "
                "bad template cell would poison that same (t, f) position for every event "
                "in the catalog, not just one, which is a worse failure mode than the "
                "catalog's own scattered gaps."
            )
        log_template = np.log10(template)

        # A lambda, not np.log10 itself: Series.apply(ufunc) dispatches the ufunc over the
        # whole Series, which fails on a Series whose elements are arrays.
        residual = raw.apply(lambda spectrum: np.log10(spectrum) - log_template)
        stacked = np.stack(residual.to_list())
        center = float(np.nanmedian(stacked))
        scale = float(_MAD_TO_SIGMA * np.nanmedian(np.abs(stacked - center)))
        if not np.isfinite(center) or not np.isfinite(scale) or scale == 0:
            raise ValueError(
                f"Degenerate spectra statistics (center={center}, scale={scale}). "
                "Check the catalog for non-positive or all-NaN spectrograms."
            )
        return cls(log_template=log_template, center=center, scale=scale)

    @classmethod
    def fit_from_catalog(cls, cfg) -> "SpectraTemplateNormalizer":
        """Fit on every spectrogram in the catalog named by ``cfg.data``, and its template.

        This is the same population ``RadioBurstDSDataset`` normalizes against at training
        time — the whole catalog, before the train/val split — so a fit here reproduces
        the constants a training run used, as long as the catalog itself has not changed.
        """
        raw = load_catalog_spectra(
            cfg.data.ds_radioburst_folder_path,
            cfg.data.ds_radioburst_index_file,
            cfg.data.ds_spectra_column,
        )
        template = load_median_template(
            cfg.data.ds_radioburst_folder_path, cfg.data.ds_spectra_template_file
        )
        return cls.fit(raw, template)

    def __call__(self, raw: pd.Series) -> pd.Series:
        """Normalize a Series of raw-flux arrays. Drop-in for ``spectra_transform=``."""

        def _normalize(spectrum: np.ndarray) -> np.ndarray:
            if spectrum.shape != self.log_template.shape:
                raise ValueError(
                    f"Spectrum shape {spectrum.shape} does not match the template's "
                    f"{self.log_template.shape}."
                )
            return (
                (np.log10(spectrum) - self.log_template - self.center) / self.scale
            ).astype(np.float32)

        return raw.apply(_normalize)

    def inverse(self, normalized) -> np.ndarray:
        """Undo the transform: normalized values back to raw flux.

        Args:
            normalized: Array (or tensor-like) of normalized values. Its trailing two
                dims broadcast against ``log_template``'s ``(T, F)`` — a single event, a
                batch ``(B, T, F)``, or a scalar-per-event ``(B, 1, 1)`` (as the linear
                baseline predicts) all work.

        Returns:
            The same shape in physical flux units, as float64 — the exponentiation
            recovers values around 1e-15, too small to keep precision in float32.
        """
        normalized = np.asarray(normalized, dtype=np.float64)
        return np.power(10.0, normalized * self.scale + self.center + self.log_template)

    def to_json(self, path) -> Path:
        """Write the constants beside a checkpoint, pinning what that run was trained on."""
        path = Path(path)
        payload = asdict(self)
        payload["log_template"] = np.asarray(self.log_template).tolist()
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    @classmethod
    def from_json(cls, path) -> "SpectraTemplateNormalizer":
        """Load constants written by :meth:`to_json`."""
        payload = json.loads(Path(path).read_text())
        payload["log_template"] = np.array(payload["log_template"], dtype=np.float32)
        return cls(**payload)


def compute_peak_amp_scale(cfg, normalizer: SpectraTemplateNormalizer) -> float:
    """Population variance of the standardized ``peak_amp`` target, over burst rows in
    the training catalog.

    ``RadioBurstMetrics(peak_amp_scale=...)`` needs this fixed once, rather than
    estimated per training batch: ``peak_amp``'s masked target is a single scalar per
    burst row, so at ``batch_size: 2`` and the catalog's typical ~44% burst rate, most
    non-empty batches carry *exactly one* burst row — too few for `.var()` to be
    anything but 0, which silently degrades the loss back to unweighted MSE in the
    thousands (see ``RadioBurstMetrics._masked_nmse``'s docstring for the floor that
    guards against blowing up, but not against this). Computing it once here, over every
    burst row in the catalog rather than one batch at a time, is what actually fixes it.

    This reads the catalog directly — the spectra CSVs and the ``burst`` column — not the
    Surya imagery, so it costs one CSV read per event rather than a full pass through the
    training dataloader (which would also download/read every NetCDF stack).

    Args:
        cfg: A ``TrainingConfig`` from ``load_radioburst_config()``.
        normalizer: Already fitted (e.g. via ``SpectraTemplateNormalizer.fit_from_catalog``)
            — must be the same one the training dataloader uses, so this reproduces the
            same standardized space the model is actually trained against.

    Returns:
        The variance of ``nanmax``-per-event ``peak_amp`` across burst rows, floored at
        1.0 for consistency with ``_masked_nmse``'s own floor (a degenerate, e.g.
        near-constant, catalog would otherwise produce a near-zero scale).
    """
    catalog = pd.read_csv(
        Path(cfg.data.ds_radioburst_folder_path) / cfg.data.ds_radioburst_index_file
    )
    raw = load_catalog_spectra(
        cfg.data.ds_radioburst_folder_path,
        cfg.data.ds_radioburst_index_file,
        cfg.data.ds_spectra_column,
    )
    normalized = normalizer(raw)
    burst_peaks = normalized[catalog["burst"] == 1].apply(np.nanmax)
    if len(burst_peaks) < 2:
        raise ValueError(
            f"Only {len(burst_peaks)} burst row(s) in the catalog; not enough to fit a "
            "population variance for peak_amp_scale."
        )
    return max(float(np.nanvar(burst_peaks.to_numpy())), 1.0)
