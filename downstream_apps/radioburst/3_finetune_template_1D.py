#!/usr/bin/env python3
"""
Train a 24 h Type II forecaster: the fine-tuned Surya model, or the linear baseline.

    # Surya (LoRA by default; see the model: section of the config)
    CUDA_VISIBLE_DEVICES=0,1 python -m downstream_apps.radioburst.3_finetune_template_1D

    # Linear baseline on per-channel image statistics
    CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.radioburst.3_finetune_template_1D --train_baseline

    # Quick sanity run: set data.max_samples: 10 in the YAML, then
    CUDA_VISIBLE_DEVICES=0 python -m downstream_apps.radioburst.3_finetune_template_1D --max-epochs 2 --no-wandb

Build the data files first (once): ``python -m downstream_apps.radioburst.prepare_data``.

All parameters live in the YAML. The CLI overrides only what varies between runs of one
config: --max-epochs, --batch-size, --s3-cache-dir and --deterministic.

The best checkpoint (lowest val_loss) is written to output.ckpt_dir as
``surya-...ckpt`` or ``baseline-...ckpt``; evaluate.py scores it on the test split.
"""

from __future__ import annotations

import argparse
import os

# Must be set BEFORE torch is imported: cuBLAS reads this once, when it initializes, so
# setting it later has no effect. Deterministic cuBLAS on CUDA >= 10.2 requires it, and
# without it every run under training.deterministic warns (or raises, when set to true).
# setdefault so a deliberate ":16:8" (smaller workspace, slightly slower) is respected.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from functools import partial
from pathlib import Path
from typing import Tuple

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from downstream_apps.radioburst.configs import TrainingConfig, load_radioburst_config
from downstream_apps.radioburst.datasets.radioburst_dataset import TypeIIDataset
from downstream_apps.radioburst.labels import LABELS
from downstream_apps.radioburst.lightning_modules.pl_simple_baseline import TypeIILightningModule
from downstream_apps.radioburst.metrics.radioburst_metrics import TypeIIMetrics
from workshop_infrastructure.assets import ensure_assets
from workshop_infrastructure.datasets.builders import build_helio_dataloaders
from workshop_infrastructure.utils import (
    apply_peft_lora,
    build_scalers,
    load_pretrained_weights,
    UploadBestCheckpointToS3,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "config_script.yaml"

# --deterministic accepts the same three tokens as the YAML key. argparse hands back a
# string, so the two boolean ones are mapped to real bools -- TrainingConfig validates
# against True/False/"warn", not against their spellings.
_DETERMINISTIC_CLI = {"false": False, "warn": "warn", "true": True}


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to the run config YAML (default: this app's config_script.yaml).")
    # Dev toggles: flip without editing the YAML
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging (useful for local runs).")
    parser.add_argument("--train_baseline", action="store_true",
                        help="Train the linear baseline instead of the Surya model.")
    # Per-job / per-machine overrides: vary across runs without touching the YAML
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override training.max_epochs from the config YAML.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override training.batch_size from the config YAML.")
    parser.add_argument("--s3-cache-dir", type=str, default=None,
                        help="Override data.s3_cache_dir (the local cache for S3 reads).")
    parser.add_argument("--deterministic", choices=tuple(_DETERMINISTIC_CLI), default=None,
                        help="Override training.deterministic. Pass 'warn' when comparing runs.")
    return parser.parse_args()


def build_datasets(cfg: TrainingConfig, scalers) -> Tuple[DataLoader, DataLoader]:
    """Train and validation DataLoaders. Only the Type II arguments below are app-specific;
    everything generic is handled by build_helio_dataloaders()."""
    return build_helio_dataloaders(
        cfg,
        TypeIIDataset,
        scalers=scalers,
        seed=cfg.seed,
        ds_events_path=cfg.data.ds_events_path,
        ds_horizon=cfg.data.ds_horizon,
        ds_negative_ratio=cfg.data.ds_negative_ratio,
        ds_exclude_behind_limb=cfg.data.ds_exclude_behind_limb,
        subsample_seed=cfg.seed,
        max_number_of_samples=cfg.data.max_samples,
    )


def build_model(cfg: TrainingConfig, scalers, run_info: dict, train_baseline: bool = False) -> L.LightningModule:
    """The model wrapped in the Lightning module. Both models output (B, 2) logits.

    ``scalers`` is only needed by the linear baseline, which reads its inputs in
    signum-log space; the Surya model works directly on normalized inputs.
    """
    metrics = {mode: TypeIIMetrics(mode) for mode in ("train_loss", "val_loss", "train_metrics", "val_metrics")}
    module = partial(TypeIILightningModule, metrics=metrics, lr=cfg.learning_rate,
                     batch_size=cfg.batch_size, run_info=run_info)

    if train_baseline:
        from downstream_apps.radioburst.models.simple_baseline import (
            LinearTypeIIModel,
            destandardize_channels,
        )
        n_features = 2 * len(cfg.data.channels) * cfg.model.time_embedding.time_dim  # mean + std
        preprocess_fn = partial(destandardize_channels, channel_order=cfg.data.channels, scalers=scalers)
        return module(LinearTypeIIModel(n_features), preprocess_fn=preprocess_fn)

    from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
    model = HelioSpectformer1D.from_config(
        cfg.model,
        num_outputs=len(LABELS),
        dtype=cfg.dtype,
        use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
    )
    load_pretrained_weights(model, cfg.model.pretrained_path)

    # Three fine-tuning regimes, selected from the model: section of the YAML:
    #   use_lora: true                          -> LoRA adapters + the whole head
    #   use_lora: false, freeze_backbone: true  -> linear probe (head only)
    #   use_lora: false, freeze_backbone: false -> full fine-tuning
    # freeze_backbone is ignored when use_lora is true: PEFT freezes every parameter,
    # then re-enables the adapters and every head_* module.
    if cfg.model.freeze_backbone:
        for name, param in model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
    if cfg.model.use_lora:
        model = apply_peft_lora(model, cfg.model.lora_config)

    _log_trainable_parameters(model)
    return module(model)


def _log_trainable_parameters(model) -> None:
    """Print the trainable/total parameter counts, so the chosen regime is visible in the log."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(f"[MODEL] Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


def build_trainer(
    cfg: TrainingConfig,
    ckpt_prefix: str,
    no_wandb: bool = False,
    max_epochs_override: int | None = None,
) -> Tuple[L.Trainer, ModelCheckpoint]:
    """Configure loggers, callbacks, and the Lightning Trainer."""
    max_epochs = max_epochs_override if max_epochs_override is not None else cfg.max_epochs

    loggers = []
    if not no_wandb:
        loggers.append(WandbLogger(
            entity=cfg.wandb_entity,  # None = personal account; set in YAML for team runs
            project=cfg.wandb_project,
            name=f"{cfg.job_id}_{ckpt_prefix}",
            log_model=False,
            save_dir=os.environ.get("TMPDIR", "./wandb/wandb_tmp"),
        ))
    loggers.append(CSVLogger("runs", name=f"{cfg.job_id}_{ckpt_prefix}"))

    Path(cfg.output.ckpt_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.output.ckpt_dir,
        filename=ckpt_prefix + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    upload_cb = UploadBestCheckpointToS3(
        checkpoint_cb=checkpoint_cb,
        bucket=cfg.output.s3_bucket,
        prefix=cfg.output.s3_prefix,
        fixed_key_name=(cfg.output.s3_best_key or None),
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        # benchmark is pinned rather than inherited: cuDNN autotuning picks algorithms by
        # timing, so leaving it on would reintroduce run-to-run drift.
        deterministic=cfg.deterministic,
        benchmark=False,
        logger=loggers,
        callbacks=[checkpoint_cb, upload_cb],
        log_every_n_steps=2,
    )
    return trainer, checkpoint_cb


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    cfg = load_radioburst_config(args.config)
    # Seeding comes after the config load, so the seed is a configured value rather than
    # a constant buried in the code. workers=True extends it to DataLoader workers.
    L.seed_everything(cfg.seed, workers=True)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.s3_cache_dir is not None:
        cfg.data.s3_cache_dir = args.s3_cache_dir
    if args.deterministic is not None:
        cfg.deterministic = _DETERMINISTIC_CLI[args.deterministic]
    # Fetch scalers, and the backbone weights unless we are training the baseline.
    ensure_assets(cfg, which=["scalers"] if args.train_baseline else ["scalers", "weights"])

    # Built once and shared: the dataset normalizes with these, and the linear baseline
    # de-standardizes with them. Two separate builds could silently disagree.
    scalers = build_scalers(info=cfg.data.scalers_path)

    train_loader, val_loader = build_datasets(cfg, scalers)
    train_labels = train_loader.dataset.labels
    print(f"[DATA] train: {len(train_labels):,} samples, positives "
          + ", ".join(f"{n}={int(train_labels[n].sum())}" for n in LABELS)
          + f" | val: {len(val_loader.dataset):,} samples")

    # Recorded with the run: evaluate.py needs it to undo the subsampling in the probabilities.
    run_info = {
        "negative_keep_fraction": train_loader.dataset.negative_keep_fraction,
        "horizon": cfg.data.ds_horizon,
        "model": "baseline" if args.train_baseline else "surya",
    }
    lit_model = build_model(cfg, scalers, run_info, train_baseline=args.train_baseline)
    trainer, checkpoint_cb = build_trainer(
        cfg,
        ckpt_prefix=run_info["model"],
        no_wandb=args.no_wandb,
        max_epochs_override=args.max_epochs,
    )

    trainer.fit(lit_model, train_loader, val_loader)

    if checkpoint_cb.best_model_path:
        print(f"[CKPT] Best checkpoint: {checkpoint_cb.best_model_path}")
        if checkpoint_cb.best_model_score is not None:
            print(f"[CKPT] Best val_loss: {float(checkpoint_cb.best_model_score):.6f}")
    else:
        print("[CKPT] No best checkpoint was saved.")


if __name__ == "__main__":
    main()
