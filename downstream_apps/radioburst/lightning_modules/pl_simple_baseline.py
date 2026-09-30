"""
pl_simple_baseline.py

The PyTorch Lightning wrapper for Type II forecasting. Used for both the linear baseline
and the fine-tuned Surya model: both map a batch to logits of shape (B, 2).

Key batch contract:
  - batch["ts"]     : input stack, (B, C, T, H, W)
  - batch["labels"] : (B, 2) float targets, in labels.LABELS order (type2, type2_ip)

Key metrics contract (the `metrics` dict passed to __init__), as in the template:
  - metrics["train_loss"] : callable(logits, labels) -> (loss_dict, weight_list). Backpropagated.
  - metrics["val_loss"]   : same form. Logged as "val_loss" — what ModelCheckpoint monitors.
                            Falls back to metrics["train_loss"] when absent.
  - metrics["train_metrics"] / metrics["val_metrics"] : per-batch extras; logged only if
                            they return anything. This app's return nothing.

Classification scores are logged once per validation epoch, from ``ValidationScores``, as
"val_metric_<label>_<score>". They are reported only and do not select checkpoints.

Component losses are logged UNWEIGHTED while "train_loss"/"val_loss" are the weighted sums,
so a component curve stays comparable across runs that used different weights.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import lightning as L
import torch

from downstream_apps.radioburst.metrics.radioburst_metrics import ValidationScores


class TypeIILightningModule(L.LightningModule):
    """
    Parameters
    ----------
    model:
        Maps a batch dict to logits of shape (B, 2).
    metrics:
        Loss/metric callables, see the module docstring.
    lr:
        Adam learning rate.
    batch_size:
        Passed to ``self.log`` for correct averaging under DDP.
    preprocess_fn:
        Optional ``(batch) -> batch`` applied before every model call, e.g.
        ``destandardize_channels`` for the linear baseline.
    run_info:
        Extra values recorded with the run (hparams.yaml and the checkpoint),
        e.g. ``negative_keep_fraction``, which evaluate.py needs to correct the
        probabilities of a model trained on subsampled negatives.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, Callable],
        lr: float,
        batch_size: Optional[int] = None,
        preprocess_fn: Optional[Callable[[Dict], Dict]] = None,
        run_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn

        self.training_loss = metrics["train_loss"]
        self.validation_loss = metrics.get("val_loss", metrics["train_loss"])
        self.training_evaluation = metrics["train_metrics"]
        self.validation_evaluation = metrics["val_metrics"]
        self.val_scores = ValidationScores()

        # An explicit dict keeps save_hyperparameters from trying to store `model`/`metrics`.
        hparams = dict(getattr(self.training_loss, "loss_weights", {}))
        hparams.update(run_info or {})
        if hparams:
            self.save_hyperparameters(hparams)

    @staticmethod
    def _combine_losses(loss_dict, weights) -> torch.Tensor:
        """Weighted sum of ``loss_dict``; ``weights`` aligned with its iteration order."""
        if not loss_dict:
            raise ValueError("loss_dict is empty; cannot compute a scalar loss.")
        return sum(loss_dict[key] * weights[n] for n, key in enumerate(loss_dict))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def _step(self, batch: Dict[str, Any], stage: str):
        labels = batch["labels"]
        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)
        logits = self(batch)

        loss_fn = self.training_loss if stage == "train" else self.validation_loss
        losses, weights = loss_fn(logits, labels)
        loss = self._combine_losses(losses, weights)

        log = dict(batch_size=self.batch_size, sync_dist=True)
        self.log(f"{stage}_loss", loss, prog_bar=True, **log)
        for key, value in losses.items():
            self.log(f"{stage}_loss_{key}", value, **log)
        # Positives per batch: at a ~5% event rate many batches have none, which is worth
        # seeing next to the loss curve.
        self.log(f"{stage}_positives", labels[:, 0].sum().float(), **log)

        evaluation = self.training_evaluation if stage == "train" else self.validation_evaluation
        extra, extra_weights = evaluation(logits, labels)
        if len(extra_weights) > 0:
            for key, value in extra.items():
                self.log(f"{stage}_metric_{key}", value, **log)
        return loss, logits, labels

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        _, logits, labels = self._step(batch, "val")
        self.val_scores.update(logits, labels)

    def on_validation_epoch_end(self) -> None:
        for key, value in self.val_scores.compute().items():
            self.log(f"val_metric_{key}", value, sync_dist=False)  # already gathered across ranks
        self.val_scores.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
