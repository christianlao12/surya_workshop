#!/usr/bin/env python3
"""
Score Type II forecasts on the held-out test split (2020-2024), against reference forecasts.

Two steps, so the expensive one runs once:

    # 1. GPU: run a checkpoint over val and test, write calibrated probabilities
    python -m downstream_apps.radioburst.evaluate predict --checkpoint checkpoints/surya-epoch=07-val_loss=0.2461.ckpt

    # 2. CPU: score every predictions file in predictions/ plus the reference forecasts
    python -m downstream_apps.radioburst.evaluate score

``score`` works with no model at all — it then scores just the reference forecasts, which
is a useful first look at how hard the task is.

What is reported, per forecast and label (on test)
--------------------------------------------------
- **TSS** (true skill statistic = hit rate - false alarm rate). A probabilistic forecast is
  thresholded at the value that maximizes TSS on **val**, never on test.
- **BSS** (Brier skill score) against climatology, i.e. the training-period event rate.
  Positive = better than always forecasting that rate. Probabilistic forecasts only.
- **ROC-AUC** and **PR-AUC**. Probabilistic forecasts only.
- A 95% interval for TSS from a **block bootstrap over solar rotations**: resampling rows
  would treat the ~20 six-hourly rows around each event as independent, and events cluster
  in active rotations, so the interval would come out far too narrow.
- **dTSS vs the M/X-flare baseline**, with a *paired* interval (both forecasts scored on the
  same resampled rotations). This is the headline number: the baseline's own interval is
  wide (~0.18-0.39 on test), so comparing two separate intervals would miss real gains.

Reference forecasts
-------------------
- ``climatology``: always the training-period event rate (the BSS reference; TSS is 0 by
  construction, so it is not listed).
- ``persistence``: 1 if a Type II (or an IP Type II, for that label) started in the past 24 h.
- ``mx_flare_24h``: 1 if an M/X flare started in the past 24 h. The one to beat.

Only numpy/pandas/sklearn are imported at module level, so ``score`` and the tests of the
scoring functions run without torch. ``predict`` imports torch when called.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from downstream_apps.radioburst.configs import load_radioburst_config
from downstream_apps.radioburst.labels import LABELS, make_labels, onset_in_past, onset_in_window
from downstream_apps.radioburst.prepare_data import ROTATION, START

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "config_script.yaml"
DEFAULT_PRED_DIR = Path("predictions")


# --------------------------------------------------------------------------------------
# Labels and reference forecasts
# --------------------------------------------------------------------------------------

def load_events(cfg) -> pd.DataFrame:
    return pd.read_csv(cfg.data.ds_events_path, parse_dates=["onset"])


def split_labels(cfg, split: str, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Labels for every issue time of a split, with the same exclusions as training."""
    path = {"train": cfg.data.train_data_path, "val": cfg.data.valid_data_path,
            "test": cfg.data.test_data_path}[split]
    index = pd.read_csv(path)
    issue = pd.to_datetime(index.loc[index["present"] == 1, "timestep"])
    events = load_events(cfg) if events is None else events
    labels = make_labels(events, issue, cfg.data.ds_horizon)
    if cfg.data.ds_exclude_behind_limb:
        hidden = onset_in_window(events.loc[events.behind_limb == 1, "onset"], labels.index, cfg.data.ds_horizon)
        labels = labels[~hidden]
    return labels


def reference_forecasts(cfg, labels: pd.DataFrame, events: pd.DataFrame, climatology: dict) -> dict:
    """Reference forecasts for the issue times of ``labels``: {name: DataFrame like labels}."""
    t, horizon = labels.index, cfg.data.ds_horizon
    flares = pd.read_csv(cfg.data.ds_flares_path, parse_dates=["start_time"])
    mx = onset_in_past(flares["start_time"], t, horizon).astype(float)
    return {
        "climatology": pd.DataFrame({n: climatology[n] for n in LABELS}, index=t),
        "persistence": pd.DataFrame({
            "type2": onset_in_past(events["onset"], t, horizon),
            "type2_ip": onset_in_past(events.loc[events.reaches_ip == 1, "onset"], t, horizon),
        }, index=t).astype(float),
        "mx_flare_24h": pd.DataFrame({n: mx for n in LABELS}, index=t),
    }


# --------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------

def correct_for_subsampling(p, negative_keep_fraction: float):
    """Undo the probability inflation from training on a fraction of the negatives.

    Keeping a fraction f of the negatives multiplies the odds the model learns by 1/f, so
    the true odds are the predicted odds times f.
    """
    p = np.asarray(p, dtype=float)
    f = negative_keep_fraction
    return p * f / (p * f + (1.0 - p))


def tss(y, forecast) -> float:
    """True skill statistic of a 0/1 forecast: hit rate minus false-alarm rate."""
    y, forecast = np.asarray(y).astype(bool), np.asarray(forecast).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    return forecast[y].mean() - forecast[~y].mean()


def best_threshold(y, p) -> float:
    """The probability threshold (forecast = p >= threshold) that maximizes TSS on (y, p)."""
    candidates = np.unique(np.asarray(p, dtype=float))
    return float(max(candidates, key=lambda c: tss(y, np.asarray(p) >= c)))


def brier_skill(y, p, p_reference) -> float:
    """1 - Brier(p) / Brier(reference). 0 = no better than the reference, 1 = perfect."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    reference = np.broadcast_to(np.asarray(p_reference, dtype=float), y.shape)
    return 1.0 - np.mean((p - y) ** 2) / np.mean((reference - y) ** 2)


def rotation_of(issue_times) -> np.ndarray:
    """Solar-rotation number of each issue time, counted from START (the bootstrap block)."""
    return ((pd.DatetimeIndex(issue_times) - START) // ROTATION).to_numpy()


def bootstrap_interval(statistic, issue_times, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of ``statistic(idx)``, resampling whole solar rotations with replacement.

    ``statistic`` receives an array of row positions (with repeats) and returns a number.
    Passing one function that scores two forecasts on the same ``idx`` gives a *paired*
    interval for their difference.
    """
    blocks = rotation_of(issue_times)
    members = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    rng = np.random.default_rng(seed)
    samples = [
        statistic(np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))]))
        for _ in range(n)
    ]
    return tuple(np.nanpercentile(samples, [2.5, 97.5]))


def score_forecast(y, p, issue_times, threshold: float | None, p_climatology: float,
                   probabilistic: bool, reference_binary, n_boot: int = 1000) -> dict:
    """All scores for one forecast of one label.

    ``threshold`` is None for 0/1 forecasts. ``reference_binary`` is the 0/1 forecast to
    compare against (the M/X-flare baseline): ``dtss`` is this forecast's TSS minus the
    reference's, with a paired rotation-bootstrap interval. An interval above 0 is the
    evidence that the forecast beats the reference.
    """
    y = np.asarray(y)
    binary = np.asarray(p) >= threshold if probabilistic else np.asarray(p).astype(bool)
    reference_binary = np.asarray(reference_binary).astype(bool)
    lo, hi = bootstrap_interval(lambda i: tss(y[i], binary[i]), issue_times, n=n_boot)
    d_lo, d_hi = bootstrap_interval(
        lambda i: tss(y[i], binary[i]) - tss(y[i], reference_binary[i]), issue_times, n=n_boot)
    row = {"tss": tss(y, binary), "tss_ci_low": lo, "tss_ci_high": hi,
           "dtss_vs_mx": tss(y, binary) - tss(y, reference_binary), "dtss_ci_low": d_lo, "dtss_ci_high": d_hi,
           "threshold": threshold, "bss": np.nan, "roc_auc": np.nan, "pr_auc": np.nan}
    if probabilistic:
        row.update(bss=brier_skill(y, p, p_climatology),
                   roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p))
    return row


# --------------------------------------------------------------------------------------
# predict (GPU)
# --------------------------------------------------------------------------------------

def strip_lightning_prefix(state_dict: dict) -> dict:
    """Keep only the wrapped model's weights, without Lightning's ``model.`` prefix."""
    return {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}


def load_model_from_checkpoint(cfg, checkpoint: dict):
    """Rebuild the model a checkpoint was trained with and load its weights (strict).

    Returns ``(model, kind)`` where kind is ``"surya"`` or ``"baseline"``. The Surya model is
    rebuilt from the config's model: section, so it must match the training config.
    """
    kind = checkpoint["hyper_parameters"]["model"]
    if kind == "baseline":
        from downstream_apps.radioburst.models.simple_baseline import LinearTypeIIModel
        model = LinearTypeIIModel(2 * len(cfg.data.channels) * cfg.model.time_embedding.time_dim)
    else:
        from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
        from workshop_infrastructure.utils import apply_peft_lora
        model = HelioSpectformer1D.from_config(
            cfg.model, num_outputs=len(LABELS), dtype=cfg.dtype,
            use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
        )
        if cfg.model.use_lora:
            model = apply_peft_lora(model, cfg.model.lora_config)
    model.load_state_dict(strip_lightning_prefix(checkpoint["state_dict"]), strict=True)
    return model.eval(), kind


def predict(cfg, checkpoint_path: Path, out_dir: Path, splits=("val", "test"), batch_size: int = 4) -> None:
    """Write ``<out_dir>/<name>_<split>.csv`` with calibrated probabilities per issue time."""
    import torch
    from functools import partial
    from torch.utils.data import DataLoader

    from downstream_apps.radioburst.datasets.radioburst_dataset import TypeIIDataset
    from downstream_apps.radioburst.models.simple_baseline import destandardize_channels
    from workshop_infrastructure.datasets.builders import _base_dataset_kwargs
    from workshop_infrastructure.utils import build_scalers

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, kind = load_model_from_checkpoint(cfg, checkpoint)
    keep_fraction = checkpoint["hyper_parameters"]["negative_keep_fraction"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    scalers = build_scalers(info=cfg.data.scalers_path)
    preprocess = partial(destandardize_channels, channel_order=cfg.data.channels, scalers=scalers) \
        if kind == "baseline" else (lambda batch: batch)
    paths = {"val": cfg.data.valid_data_path, "test": cfg.data.test_data_path}
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(checkpoint_path).stem.split("-")[0]  # "surya" or "baseline"

    for split in splits:
        dataset = TypeIIDataset(
            index_path=paths[split], phase="val",  # never subsampled
            ds_events_path=cfg.data.ds_events_path, ds_horizon=cfg.data.ds_horizon,
            ds_exclude_behind_limb=cfg.data.ds_exclude_behind_limb,
            max_number_of_samples=cfg.data.max_samples,
            **_base_dataset_kwargs(cfg, scalers),
        )
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=cfg.num_workers, shuffle=False,
                            multiprocessing_context="spawn" if cfg.num_workers > 0 else None)
        times, probs = [], []
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            for batch in loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                probs.append(torch.sigmoid(model(preprocess(batch)).float()).cpu().numpy())
                times += list(batch["ds_index"])
        p = correct_for_subsampling(np.concatenate(probs), keep_fraction)
        out = pd.DataFrame(p, columns=list(LABELS), index=pd.to_datetime(times))
        out.index.name = "issue_time"
        out.to_csv(out_dir / f"{name}_{split}.csv")
        print(f"[PREDICT] {split}: {len(out):,} issue times -> {out_dir / f'{name}_{split}.csv'}")


# --------------------------------------------------------------------------------------
# score (CPU)
# --------------------------------------------------------------------------------------

def score(cfg, pred_dir: Path, n_boot: int = 1000) -> pd.DataFrame:
    """Score every ``<name>_test.csv`` in ``pred_dir`` plus the reference forecasts."""
    events = load_events(cfg)
    train, val, test = (split_labels(cfg, s, events) for s in ("train", "val", "test"))
    climatology = {n: float(train[n].mean()) for n in LABELS}

    references = reference_forecasts(cfg, test, events, climatology)
    forecasts = {k: (v, None) for k, v in references.items() if k != "climatology"}
    for test_file in sorted(Path(pred_dir).glob("*_test.csv")):
        name = test_file.stem[: -len("_test")]
        p_test = pd.read_csv(test_file, index_col=0, parse_dates=True)
        p_val = pd.read_csv(test_file.with_name(f"{name}_val.csv"), index_col=0, parse_dates=True)
        forecasts[name] = (p_test, p_val)

    rows = []
    for name, (p_test, p_val) in forecasts.items():
        common = test.index.intersection(p_test.index)
        for label in LABELS:
            probabilistic = p_val is not None
            threshold = None
            if probabilistic:
                val_common = val.index.intersection(p_val.index)
                threshold = best_threshold(val.loc[val_common, label], p_val.loc[val_common, label])
            row = score_forecast(test.loc[common, label], p_test.loc[common, label], common, threshold,
                                 climatology[label], probabilistic,
                                 reference_binary=references["mx_flare_24h"].loc[common, label], n_boot=n_boot)
            rows.append({"forecast": name, "label": label, "n": len(common),
                         "event_rate": float(test.loc[common, label].mean()), **row})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR),
                        help="Where predictions are written/read (default: ./predictions).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="GPU: checkpoint -> calibrated val/test probabilities.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    s = sub.add_parser("score", help="CPU: score predictions and reference forecasts on test.")
    s.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_radioburst_config(args.config)
    if args.command == "predict":
        predict(cfg, Path(args.checkpoint), Path(args.pred_dir), batch_size=args.batch_size)
    else:
        table = score(cfg, Path(args.pred_dir), n_boot=args.n_boot)
        with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
            print(table.to_string(index=False))
        Path(args.pred_dir).mkdir(parents=True, exist_ok=True)
        table.to_csv(Path(args.pred_dir) / "scores.csv", index=False)
        print(f"\nsaved {Path(args.pred_dir) / 'scores.csv'}")


if __name__ == "__main__":
    main()
