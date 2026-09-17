"""
Run a trained radio-burst model on one timestamp.

Training answers "how well does this fit?"; this module answers "what does the model say
about 15:00 on Tuesday?". Three things stand between a saved checkpoint and that answer,
and each gets one function here:

1. **A timestamp is not a frame.** Surya frames sit on a strict 12-minute grid and some
   are missing, so a requested time has to be snapped and checked — see
   :func:`resolve_frame`.
2. **The dataset reads a CSV, not a timestamp.** ``HelioNetCDFDataset`` takes an index
   file, so one frame becomes a one-row index — see :func:`single_frame_index`.
3. **The checkpoint is LoRA-wrapped.** Its keys carry the ``model.base_model.model.``
   prefix that PyTorch Lightning and PEFT add, so the model must be rebuilt and wrapped
   the same way before the weights will load — see :func:`load_finetuned_burst_model`.

What the prediction means: with ``ds_forecast_horizon: 3h`` and 1-hour catalog windows, a
frame at time ``t`` forecasts a burst in ``[t + 3h, t + 4h]``, and the predicted
spectrogram spans those 60 minutes. Spectrogram values come back in the normalized space
of ``spectra_transform`` — use :class:`~downstream_apps.radioburst.spectra_transform.SpectraLog10Normalizer.inverse`
to read them as flux.
"""

from __future__ import annotations

import shutil
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from downstream_apps.radioburst.models.finetune_burst import HelioSpectformerBurst
from downstream_apps.radioburst.models.simple_baseline import (
    TwoStageBurstModel,
    destandardize_channels,
)
from downstream_apps.radioburst.spectra_transform import SpectraLog10Normalizer
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset

# The config -> HelioNetCDFDataset argument mapping lives in workshop_infrastructure and is
# reused rather than repeated here: the public builders always return a train/val pair from
# the two index paths in the config, which is not what single-frame inference needs.
from workshop_infrastructure.datasets.builders import _base_dataset_kwargs
from workshop_infrastructure.utils import apply_peft_lora, build_scalers

# Cadence of the Surya index. Every reference timestamp must land exactly on this grid.
SURYA_CADENCE = "12min"

# Prefixes the training stack adds around the model's own parameter names.
LIGHTNING_PREFIX = "model."
PEFT_MARKER = "base_model.model."


def latest_checkpoint(ckpt_dir, pattern: str = "finetune-*.ckpt") -> Path:
    """Return the most recently modified checkpoint in ``ckpt_dir``.

    Args:
        ckpt_dir: Directory holding the checkpoints (``cfg.output.ckpt_dir``).
        pattern: Glob for the checkpoints to consider. The default matches the
            fine-tuning runs and skips the linear baseline's ``baseline-*.ckpt``.

    Raises:
        FileNotFoundError: If nothing matches, naming the directory searched.
    """
    matches = sorted(Path(ckpt_dir).glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matching {pattern!r} in {Path(ckpt_dir).resolve()}. "
            "Train a model first (2_finetune_template_1D.ipynb), or pass ckpt_path explicitly."
        )
    return matches[-1]


def resolve_frame(timestamp, index_path, cadence: str = SURYA_CADENCE) -> pd.Series:
    """Find the Surya frame to feed the model for a requested time.

    The requested time is rounded to the nearest point on the index's fixed cadence, then
    required to be present. Nothing is interpolated: a missing frame is a missing
    observation, and silently substituting a neighbour would change what the prediction is
    about.

    Args:
        timestamp: Anything ``pd.Timestamp`` accepts, e.g. ``"2014-01-21 02:00"``.
        index_path: A Surya index CSV with ``path``, ``timestep`` and ``present`` columns.
        cadence: Grid to snap to. The Surya index is 12-minutely.

    Returns:
        The matching index row, with ``path`` and ``timestep``.

    Raises:
        ValueError: If that frame is not available, naming the nearest usable frames
            either side so the caller can pick one.
    """
    index = pd.read_csv(index_path, usecols=["path", "timestep", "present"])
    index["timestep"] = pd.to_datetime(index["timestep"])
    available = index[index["present"] == 1].sort_values("timestep").reset_index(drop=True)

    wanted = pd.Timestamp(timestamp).round(cadence)
    match = available[available["timestep"] == wanted]
    if len(match):
        return match.iloc[0]

    earlier = available[available["timestep"] < wanted]["timestep"]
    later = available[available["timestep"] > wanted]["timestep"]
    nearest = [str(earlier.iloc[-1]) if len(earlier) else None,
               str(later.iloc[0]) if len(later) else None]
    raise ValueError(
        f"No usable Surya frame at {wanted} (requested {timestamp}) in {index_path}.\n"
        f"The index covers {available['timestep'].iloc[0]} to {available['timestep'].iloc[-1]}.\n"
        f"Nearest usable frames: before={nearest[0]}, after={nearest[1]}."
    )


@contextmanager
def single_frame_index(frame: pd.Series, out_dir=None):
    """Write the one-row index CSV that makes a length-1 dataset, and clean it up after.

    Args:
        frame: A row from :func:`resolve_frame`.
        out_dir: Where to write it. A temporary directory by default, removed on exit;
            pass a path to keep the file for inspection.

    Yields:
        Path to the written CSV.
    """
    temporary = out_dir is None
    out_dir = Path(tempfile.mkdtemp(prefix="surya_infer_")) if temporary else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "single_frame_index.csv"
    pd.DataFrame(
        [{"path": frame["path"], "timestep": frame["timestep"], "present": 1}]
    ).to_csv(path, index=False)
    try:
        yield path
    finally:
        if temporary:
            shutil.rmtree(out_dir, ignore_errors=True)


def build_single_frame_dataset(cfg, index_path, scalers=None) -> HelioNetCDFDataset:
    """Build a dataset over one index, with the same settings training used.

    Plain ``HelioNetCDFDataset``, not ``RadioBurstDSDataset``: the backbone reads only
    ``ts`` and ``time_delta_input``, and inference has no label to align, so the burst
    catalog is not involved in building the input.

    Args:
        cfg: A ``TrainingConfig`` from ``load_radioburst_config()``.
        index_path: Index CSV, e.g. from :func:`single_frame_index`.
        scalers: Normalization statistics; built from the config if omitted.

    Returns:
        The dataset. ``phase="val"`` because inference is never the training phase.
    """
    if scalers is None:
        scalers = build_scalers(info=cfg.data.scalers_path)
    return HelioNetCDFDataset(
        index_path=str(index_path),
        phase="val",
        load_forecast_frames=False,
        **_base_dataset_kwargs(cfg, scalers),
    )


def _strip_lightning_prefix(state_dict: dict) -> dict:
    """Drop the ``model.`` that ``LightningModule.model`` adds to every key."""
    if not all(k.startswith(LIGHTNING_PREFIX) for k in state_dict):
        return state_dict
    return {k[len(LIGHTNING_PREFIX):]: v for k, v in state_dict.items()}


def head_geometry(state_dict: dict) -> tuple[int, int]:
    """Read the head's shape out of a checkpoint instead of hardcoding it.

    ``head_spectra`` is a rank-r bottleneck: its first layer projects the embedding down to
    ``spectra_rank`` coefficients and its second maps those onto a flattened spectrogram.

    Args:
        state_dict: A checkpoint state dict, Lightning prefix already stripped.

    Returns:
        ``(spectra_rank, n_spectrum_values)`` where the second is ``T * F``.

    Raises:
        KeyError: If the head tensors are absent — the checkpoint is not a
            ``HelioSpectformerBurst``, or predates the ``head_`` naming convention.
    """
    def find(suffix: str) -> torch.Tensor:
        keys = [k for k in state_dict if k.endswith(suffix)]
        if not keys:
            raise KeyError(
                f"No tensor ending in {suffix!r} in this checkpoint, so it does not match "
                "the current HelioSpectformerBurst head. Either it is not a burst "
                "checkpoint, or it was trained before the head was last changed — in which "
                "case retrain, or check out the code that produced it."
            )
        return state_dict[keys[0]]

    spectra_rank = find("head_spectra.modules_to_save.default.0.weight").shape[0]
    n_values = find("head_spectra.modules_to_save.default.1.weight").shape[0]
    return spectra_rank, n_values


def load_finetuned_burst_model(
    cfg,
    ckpt_path,
    spectrum_shape: tuple[int, int] | None = None,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    """Rebuild the fine-tuned model and load a training checkpoint into it.

    The checkpoint stores a ``LightningModule`` wrapping a PEFT-wrapped model, so its keys
    look like ``model.base_model.model.…``. Reloading therefore means rebuilding the same
    architecture, applying LoRA the same way, then loading strictly — a strict load is
    what catches an architecture that does not match the checkpoint, instead of silently
    leaving parts of the model at their random initialization.

    The pretrained Surya backbone (~1.8 GB) is deliberately *not* loaded first: every
    parameter is about to be overwritten by the strict load, so reading it would only cost
    time.

    Args:
        cfg: A ``TrainingConfig`` from ``load_radioburst_config()``.
        ckpt_path: The ``.ckpt`` written by ``ModelCheckpoint``.
        spectrum_shape: ``(T, F)`` of the predicted spectrogram. Inferred from the
            checkpoint and the catalog's median template when omitted.
        device: Where to put the model. CUDA when available, else CPU.

    Returns:
        The model in ``eval()`` mode on ``device``.
    """
    device = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _strip_lightning_prefix(checkpoint["state_dict"])
    spectra_rank, n_values = head_geometry(state_dict)

    if spectrum_shape is None:
        spectrum_shape = _spectrum_shape_from_template(cfg, n_values)
    if int(np.prod(spectrum_shape)) != n_values:
        raise ValueError(
            f"spectrum_shape {tuple(spectrum_shape)} has {int(np.prod(spectrum_shape))} "
            f"values, but the checkpoint's decoder outputs {n_values}."
        )

    model = HelioSpectformerBurst.from_config(
        cfg.model,
        spectrum_shape=tuple(spectrum_shape),
        spectra_rank=spectra_rank,
        dtype=cfg.dtype,
        use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
    )
    if any(PEFT_MARKER in key for key in state_dict):
        model = apply_peft_lora(model, cfg.model.lora_config)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def _spectrum_shape_from_template(cfg, n_values: int) -> tuple[int, int]:
    """Derive ``(T, F)`` from the catalog's median template, which has the target's shape."""
    template_path = Path(cfg.data.ds_radioburst_folder_path) / cfg.data.ds_spectra_template_file
    if not cfg.data.ds_spectra_template_file or not template_path.exists():
        raise ValueError(
            f"Cannot infer spectrum_shape: no median template at {template_path}. "
            "Pass spectrum_shape=(T, F) explicitly."
        )
    n_timesteps = len(pd.read_csv(template_path))
    if n_values % n_timesteps:
        raise ValueError(
            f"The checkpoint's {n_values} spectrogram values do not divide by the "
            f"template's {n_timesteps} timesteps. Pass spectrum_shape=(T, F) explicitly."
        )
    return n_timesteps, n_values // n_timesteps


@torch.no_grad()
def predict(model: torch.nn.Module, batch: dict, device: str | torch.device | None = None) -> dict:
    """Run one batch through the model.

    Args:
        model: A model from :func:`load_finetuned_burst_model`.
        batch: A batch dict with at least ``ts`` and ``time_delta_input``.
        device: Defaults to the device the model is already on.

    Returns:
        ``burst_probability`` ``(B,)`` — the sigmoid of the burst logit — and ``spectra``
        ``(B, T, F)`` in normalized space. Compare the probability against 0.5, the
        threshold ``torchmetrics.F1Score(task="binary")`` uses during training.
    """
    if device is None:
        device = next(model.parameters()).device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    output = model(batch)
    return {
        "burst_probability": torch.sigmoid(output["burst_logit"]).reshape(-1).float().cpu().numpy(),
        "spectra": output["spectra"].float().cpu().numpy(),
    }


def load_baseline_model(cfg, ckpt_path, device: str | torch.device | None = None) -> TwoStageBurstModel:
    """Rebuild the linear baseline from a training checkpoint.

    Like the Surya loader, the model's shape is read out of the checkpoint rather than
    rebuilt from the config: ``classifier.weight`` gives the flattened input width
    (``2 * n_channels * n_timesteps``) and the ``normalized_template`` buffer gives the
    spectrogram's ``(T, F)``. The template is therefore restored from the checkpoint too,
    so reloading does not depend on the catalog's template CSV still being there, or still
    holding what it held during training.

    Args:
        cfg: Unused today, accepted so this matches ``load_finetuned_burst_model``'s shape
            and stays the obvious place to add config-driven behavior.
        ckpt_path: The ``baseline-*.ckpt`` written by ``ModelCheckpoint``.
        device: Where to put the model. CPU by default — the baseline is two linear layers.

    Returns:
        The model in ``eval()`` mode on ``device``.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _strip_lightning_prefix(checkpoint["state_dict"])
    try:
        input_dim = state_dict["classifier.weight"].shape[1]
        spectrum_shape = tuple(state_dict["normalized_template"].shape)
    except KeyError as missing:
        raise KeyError(
            f"{missing} is not in this checkpoint, so it is not a TwoStageBurstModel. "
            "Did you pass a fine-tuning checkpoint (finetune-*.ckpt) by mistake?"
        ) from None

    model = TwoStageBurstModel(input_dim, np.zeros(spectrum_shape, dtype=np.float32))
    model.load_state_dict(state_dict, strict=True)  # also restores normalized_template
    return model.to(device).eval()


@torch.no_grad()
def predict_baseline(model, batch: dict, cfg, scalers, device=None) -> dict:
    """Run one batch through the linear baseline.

    The baseline reads its inputs in **signum-log** space, not the normalized space the
    dataset produces, so this applies ``destandardize_channels()`` first — the same
    ``preprocess_fn`` ``RadioBurstLightningModule`` was given during training. Skipping it
    silently feeds the model inputs it never saw.

    Args:
        model: A model from :func:`load_baseline_model`.
        batch: A batch dict with at least ``ts``.
        cfg: Supplies ``cfg.data.channels``, the channel order of ``ts``.
        scalers: The same normalization statistics the dataset was built with.
        device: Defaults to the device the model is already on.

    Returns:
        ``burst_probability`` ``(B,)`` — already a sigmoid inside the model, unlike the
        Surya head's logit — ``peak_amp`` ``(B,)`` and ``spectra`` ``(B, T, F)``. The
        spectrogram is ``peak_amp`` times a fixed template, in **raw flux units**: the
        baseline is trained against untransformed targets, so nothing has to be inverted.
    """
    if device is None:
        device = next(model.parameters()).device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    output = model(destandardize_channels(batch, cfg.data.channels, scalers))
    return {
        "burst_probability": output["burst_prob"].reshape(-1).float().cpu().numpy(),
        "peak_amp": output["peak_amp"].reshape(-1).float().cpu().numpy(),
        "spectra": output["spectra"].float().cpu().numpy(),
    }


def find_catalog_window(cfg, window_start, tolerance=None) -> pd.Series | None:
    """The catalog window a forecast would have been scored against, if there is one.

    This reproduces the pairing ``RadioBurstDSDataset`` does at training time, so the
    notebook never shows a "ground truth" that training itself would not have matched to
    this frame. Written out, the training rule pairs a frame ``t`` with the earliest
    catalog window whose ``window_start - ds_forecast_horizon`` is at or after ``t``, and
    within ``ds_time_tolerance``. Since ``window_start = t + horizon``, the horizon cancels:
    it is the earliest catalog window at or after ``window_start``, within the tolerance.

    Args:
        cfg: A ``TrainingConfig`` from ``load_radioburst_config()``.
        window_start: Start of the forecast window — the frame's timestamp plus
            ``ds_forecast_horizon``, not the raw frame timestamp.
        tolerance: Maximum gap allowed. Defaults to ``cfg.data.ds_time_tolerance``, the
            tolerance training enforced.

    Returns:
        The catalog row, with its ``burst`` label and spectra filename, or ``None`` when
        no window matches. ``None`` is the ordinary case, not an error: the catalog holds
        170 windows across 14 years, so most timestamps have nothing to compare against.
    """
    tolerance = pd.Timedelta(cfg.data.ds_time_tolerance if tolerance is None else tolerance)
    window_start = pd.Timestamp(window_start)

    catalog = pd.read_csv(
        Path(cfg.data.ds_radioburst_folder_path) / cfg.data.ds_radioburst_index_file
    )
    times = pd.to_datetime(catalog[cfg.data.ds_time_column])

    forward = catalog[times >= window_start]
    if forward.empty:
        return None
    row = forward.loc[times[times >= window_start].idxmin()]
    if pd.Timestamp(row[cfg.data.ds_time_column]) - window_start > tolerance:
        return None
    return row


SPECTRA_NORM_FILENAME = "spectra_norm.json"


def load_spectra_normalizer(cfg, ckpt_path=None) -> SpectraLog10Normalizer:
    """Get the constants needed to read a prediction as flux.

    Prefers the ``spectra_norm.json`` a training run saved next to its checkpoint, because
    that pins exactly what the model was trained against. Falls back to re-fitting on the
    catalog, which reproduces those constants **as long as the catalog has not changed
    since training** — the fallback warns, because a changed catalog silently rescales
    every prediction rather than raising.

    Args:
        cfg: A ``TrainingConfig`` from ``load_radioburst_config()``.
        ckpt_path: The checkpoint being used; its folder is searched for the sidecar.

    Returns:
        The normalizer, with :meth:`SpectraLog10Normalizer.inverse` ready to use.
    """
    if ckpt_path is not None:
        sidecar = Path(ckpt_path).parent / SPECTRA_NORM_FILENAME
        if sidecar.exists():
            return SpectraLog10Normalizer.from_json(sidecar)

    warnings.warn(
        f"No {SPECTRA_NORM_FILENAME} beside the checkpoint — re-fitting the spectra "
        "normalization on the catalog. This matches training only if "
        f"{cfg.data.ds_radioburst_index_file} and its spectra files are unchanged since "
        "that run; if they are not, predicted flux values are silently rescaled.",
        stacklevel=2,
    )
    return SpectraLog10Normalizer.fit_from_catalog(cfg)


def spectrogram_axes(template_path) -> tuple[np.ndarray, np.ndarray]:
    """Axes for plotting a predicted spectrogram, read from the median template.

    The template mirrors the target's grid: a leading ``minutes_from_start`` column and one
    column per frequency bin, named ``"<freq>_MHz"``. 38 of those frequencies repeat (the
    RAD2 receiver pairs bins), and pandas suffixes repeated names (``"13.800000_MHz.1"``),
    so the frequency is parsed off the front of the label.

    Args:
        template_path: Path to the median burst-spectrogram template CSV.

    Returns:
        ``(minutes, frequencies_mhz)`` — lengths T and F respectively.
    """
    template = pd.read_csv(template_path)
    minutes = template.iloc[:, 0].to_numpy(dtype=np.float64)
    frequencies = np.array(
        [float(str(col).split("_")[0]) for col in template.columns[1:]], dtype=np.float64
    )
    return minutes, frequencies
