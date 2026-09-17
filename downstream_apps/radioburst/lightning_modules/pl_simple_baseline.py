"""
pl_simple_baseline.py

A minimal PyTorch Lightning wrapper for training a radio burst prediction model.

This module defines a single LightningModule (RadioBurstLightningModule) that:
  - Calls a user-provided PyTorch model on batched inputs (batch["ts"])
  - Computes one or more training/validation losses via a user-provided loss function
  - Logs scalar losses and evaluation metrics using Lightning's built-in logging
  - Configures a simple Adam optimizer

Intended use:
  - Provide a clean, readable baseline training loop in Lightning
  - Separate "model architecture" from "training mechanics"
  - Demonstrate how to log multiple losses/metrics consistently

Key batch contract:
  - batch["ts"]          : torch.Tensor input stack (e.g., [B, C, T, H, W])
  - batch["burst"]       : torch.Tensor 0/1 burst label (e.g., [B])
  - batch["diagnostics"] : torch.Tensor peak_amp regression target, shape [B, 1]
                            (narrowed via ds_diagnostics_columns: [peak_amp] in the
                            config); NaN in rows where burst == 0. Compared against the
                            linear baseline's "peak_amp" output by RadioBurstMetrics.
  - batch["spectra"]     : torch.Tensor radio spectrogram target, shape [B, T, F].
                            Compared against HelioSpectformerBurst's "spectra" output by
                            RadioBurstSpectraMetrics.

All three targets are passed to the metrics in one dict; each metrics class reads only the
keys its model predicts.

Optional preprocessing:
  - If ``preprocess_fn`` is provided to ``__init__``, it is called on the batch dict
    before every model call. This is the intended hook for input transformations (such
    as inverse-normalizing SDO channels) that should not live inside the model.

Key metrics contract (the `metrics` dict passed to __init__):
  - metrics["train_loss"]    : callable(output, target) -> (loss_dict, weight_list)
        Backpropagated. Logged as "train_loss".
  - metrics["val_loss"]      : callable(output, target) -> (loss_dict, weight_list)
        Optional. Logged as "val_loss" and therefore what ModelCheckpoint monitors.
        Falls back to metrics["train_loss"] when absent.
  - metrics["train_metrics"] : callable(output, target) -> (metric_dict, weight_list)
  - metrics["val_metrics"]   : callable(output, target) -> (metric_dict, weight_list)
        Reported only. These do NOT affect checkpoint selection — "val_loss" does.

Where:
  - loss_dict / metric_dict map string names -> torch scalar tensors
  - weight_list is a list-like of floats (or tensors) aligned with the dict iteration order
    used by this baseline to form a weighted sum loss.

Two consequences of that weighted sum worth knowing when reading the logs:

  - Component losses are logged as they come back from the metrics object, i.e. UNWEIGHTED,
    while "train_loss"/"val_loss" are the weighted sums. At any weight other than 1.0 the
    components will not add up to the total. This is deliberate: a raw component curve stays
    comparable across runs that used different weights.
  - The weights themselves are constructor arguments of the metrics object and appear in no
    config file, so this module reads them off ``metrics["train_loss"].loss_weights`` (when
    present) and records them as hyperparameters — otherwise a finished run would carry no
    record of the weighting that produced it.

This module also logs "train_burst_rows"/"val_burst_rows", the number of burst rows per
batch. The regression terms are masked to those rows, so a batch with none contributes a
flat 0.0 to that term; the counter is what makes that visible instead of looking like a
perfectly fitted batch.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import lightning as L
import torch


# Type aliases for clarity in documentation / teaching.
LossDict = Mapping[str, torch.Tensor]
MetricDict = Mapping[str, torch.Tensor]
Weights = Any  # often a list[float] or list[torch.Tensor]


class RadioBurstLightningModule(L.LightningModule):
    """
    PyTorch LightningModule for radio-burst prediction training.

    This class wraps:
      (1) a user-provided PyTorch model (nn.Module-like) and
      (2) a set of loss/metric callables packaged in the `metrics` dictionary.

    Parameters
    ----------
    model:
        A callable model (typically torch.nn.Module) that accepts the batch input tensor
        `x = batch["ts"]` and returns predictions `output`.

    metrics:
        Dictionary containing the training loss function and metric functions.

        Required keys:
          - "train_loss": callable(output, target) -> (losses, weights)
              losses: dict[str, torch.Tensor] scalar losses
              weights: list-like aligned with iteration order of losses.keys()
          - "train_metrics": callable(output, target) -> (metrics, weights)
          - "val_metrics": callable(output, target) -> (metrics, weights)

        Optional key:
          - "val_loss": callable(output, target) -> (losses, weights)
              The validation objective. Defaults to "train_loss" when not supplied, so
              older metrics dicts keep working unchanged.

        The module uses:
          - train_loss in training_step, backpropagated and logged as "train_loss"
          - val_loss in validation_step, logged as "val_loss" — the quantity
            ModelCheckpoint monitors
          - train_metrics logged during training_step (if weights is non-empty)
          - val_metrics logged during validation_step (if weights is non-empty).
            Reported only; they do not influence checkpoint selection.

        If metrics["train_loss"] exposes a ``loss_weights`` mapping, it is recorded as this
        module's hyperparameters (see the module docstring).

    lr:
        Learning rate for the Adam optimizer.

    batch_size:
        Optional batch size passed to Lightning's `self.log(..., batch_size=...)`.
        This improves correct averaging behavior when using distributed settings
        or variable batch sizes.

    preprocess_fn:
        Optional callable applied to the batch dict before every model call.
        Signature: ``(batch: dict) -> dict``. Use this to apply input
        transformations (e.g., ``destandardize_channels``) without
        embedding them in the model itself.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, Callable[..., Tuple[Dict[str, torch.Tensor], Weights]]],
        lr: float,
        batch_size: Optional[int] = None,
        preprocess_fn: Optional[Callable[[Dict], Dict]] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.model = model
        self.preprocess_fn = preprocess_fn

        # Loss callables: return (loss_dict, weight_list)
        self.training_loss = metrics["train_loss"]
        # "val_loss" is optional: falling back to train_loss keeps a metrics dict written
        # before this key existed working, with identical behavior.
        self.validation_loss = metrics.get("val_loss", metrics["train_loss"])

        # Metric callables: return (metric_dict, weight_list)
        self.training_evaluation = metrics["train_metrics"]
        self.validation_evaluation = metrics["val_metrics"]

        self.lr = lr

        # Record the loss weights with the run. Passing an explicit dict keeps
        # save_hyperparameters from introspecting this signature (which would try to store
        # `model` and `metrics`); the values reach the WandB config, the CSVLogger's
        # hparams.yaml, and the saved checkpoint, so a checkpoint knows what it was trained
        # under. getattr, because a metrics object need not expose the property.
        loss_weights = getattr(self.training_loss, "loss_weights", {})
        if loss_weights:
            self.save_hyperparameters(dict(loss_weights))

    @staticmethod
    def _combine_losses(loss_dict: LossDict, weights: Weights) -> torch.Tensor:
        """Return a weighted sum of the losses in ``loss_dict``.

        ``weights`` must be aligned with ``loss_dict.keys()`` iteration order.
        Raises ``ValueError`` if ``loss_dict`` is empty.
        """
        loss = None
        for n, key in enumerate(loss_dict.keys()):
            component = loss_dict[key] * weights[n]
            loss = component if loss is None else (loss + component)
        if loss is None:
            raise ValueError("loss_dict is empty; cannot compute a scalar loss.")
        return loss

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass used by Lightning and by explicit calls in steps.

        Parameters
        ----------
        batch:
            Batch dict (at minimum contains ``"ts"``, ``"burst"``, and ``"diagnostics"``).

        Returns
        -------
        torch.Tensor
            Model predictions for the batch.
        """
        return self.model(batch)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Runs one training step on a single batch.

        Workflow
        --------
        1) Extract inputs and targets from the batch:
              x = batch["ts"]
              target = {"burst": batch["burst"], "diagnostics": batch["diagnostics"], "spectra": batch["spectra"]}
        2) Compute model output:
              output = self(batch)
        3) Compute per-component losses and combine via provided weights:
              training_losses, training_loss_weights = training_loss(output, target)
        4) Log:
              - total weighted loss as "train_loss" (progress bar)
              - each component loss as "train_loss_<name>", unweighted
              - the batch's burst-row count as "train_burst_rows"
              - training metrics as "train_metric_<name>" (if any)

        Notes
        -----
        - `target` is a dict (`"burst"`, `"diagnostics"`, `"spectra"`); `output` is the
          model's dict. Loss/metric callables index into both by key rather than operating
          on a single tensor: `RadioBurstMetrics` compares `output["peak_amp"]` with
          `target["diagnostics"]` (linear baseline), `RadioBurstSpectraMetrics` compares
          `output["spectra"]` with `target["spectra"]` (Surya fine-tuning head).
        - The loss combination depends on dict iteration order; ensure loss dict
          insertion order is consistent if that matters.

        Returns
        -------
        torch.Tensor
            The scalar training loss used for backpropagation.
        """
        target = {"burst": batch["burst"], "diagnostics": batch["diagnostics"], "spectra": batch["spectra"]}

        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)
        output = self(batch)
        training_losses, training_loss_weights = self.training_loss(output, target)
        loss = self._combine_losses(training_losses, training_loss_weights)

        # Log aggregate loss and component losses. Components are the raw, unweighted
        # values the metrics object returned; `loss` is the weighted sum of them.
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        # Burst rows in this batch. The masked regression term is 0.0 when this is 0, so the
        # epoch mean of that term is pulled toward zero by burst-free batches; this is how
        # you tell that apart from the decoder actually improving.
        self.log("train_burst_rows", target["burst"].sum().float(), prog_bar=False, batch_size=self.batch_size, sync_dist=True)
        for key in training_losses.keys():
            self.log(f"train_loss_{key}", training_losses[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        # Log evaluation metrics (optional).
        training_evaluation_metrics, training_evaluation_weights = self.training_evaluation(output, target)
        if len(training_evaluation_weights) > 0:
            for key in training_evaluation_metrics.keys():
                self.log(f"train_metric_{key}", training_evaluation_metrics[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        Runs one validation step on a single batch.

        Workflow
        --------
        1) Extract inputs and targets
        2) Compute output
        3) Compute validation losses and combine via weights
        4) Log:
              - total weighted loss as "val_loss" (progress bar)
              - each component loss as "val_loss_<name>", unweighted
              - the batch's burst-row count as "val_burst_rows"
              - validation metrics as "val_metric_<name>" (if any)

        Notes
        -----
        - The loss is computed with `self.validation_loss`, which comes from
          metrics["val_loss"] and falls back to metrics["train_loss"] when that key is
          absent. Supply a distinct "val_loss" callable to monitor something other than
          the training objective.
        - "val_loss" is what ModelCheckpoint monitors. The `val_metrics` logged at the end
          of this method are reported only and do not affect checkpoint selection.
        - No value is returned (Lightning uses logs for validation tracking).
        """
        target = {"burst": batch["burst"], "diagnostics": batch["diagnostics"], "spectra": batch["spectra"]}

        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)
        output = self(batch)
        val_losses, val_loss_weights = self.validation_loss(output, target)
        loss = self._combine_losses(val_losses, val_loss_weights)

        # Log aggregate loss and component losses (components raw; `loss` is the weighted sum).
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log("val_burst_rows", target["burst"].sum().float(), prog_bar=False, batch_size=self.batch_size, sync_dist=True)
        for key in val_losses.keys():
            self.log(f"val_loss_{key}", val_losses[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        # Log evaluation metrics (optional).
        val_evaluation_metrics, val_evaluation_weights = self.validation_evaluation(output, target)
        if len(val_evaluation_weights) > 0:
            for key in val_evaluation_metrics.keys():
                self.log(f"val_metric_{key}", val_evaluation_metrics[key], prog_bar=False, batch_size=self.batch_size, sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer used by Lightning.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer over all module parameters with learning rate `self.lr`.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
