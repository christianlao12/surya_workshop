"""
Metrics for the radio-burst downstream app.

Two classes, one per model:
- RadioBurstMetrics         — for the linear baseline (TwoStageBurstModel): burst probability
                              plus a scalar ``peak_amp``.
- RadioBurstSpectraMetrics  — for the Surya fine-tuning head (HelioSpectformerBurst): burst
                              logit plus a full (T, F) spectrogram.

Both define four metric sets:
- "train_loss"    — differentiable loss that drives backpropagation (burst BCE + masked MSE).
- "val_loss"      — the quantity logged as `val_loss` and used to select checkpoints.
                    Defaults to the same loss as "train_loss"; override it when your task
                    needs a different validation objective.
- "train_metrics" — non-differentiable metrics logged during training (F1 + RRSE).
- "val_metrics"   — metrics logged at validation for reporting only (F1 + MSE + RRSE).
                    These do NOT influence checkpoint selection — "val_loss" does.

The __call__ method selects the appropriate metric set based on the mode passed at
construction time. The dictionary keys returned by each method become the metric names
propagated to the logger (e.g. WandB, CSV).

The regression terms are masked to burst rows (``target["burst"] == 1``): quiet windows
only train the classifier. Non-finite target cells (missing measurements) are skipped too.

RadioBurstMetrics' regression target is a single scalar, ``peak_amp`` — the amplitude
TwoStageBurstModel regresses to scale its median spectrogram template
(``downstream_apps/radioburst/models/simple_baseline.py``). It arrives as
``target["diagnostics"]``, shape ``(B, 1)``: that key name is owned by
``RadioBurstDSDataset.__getitem__``, but it only holds ``peak_amp``, by convention of
``ds_diagnostics_columns: [peak_amp]`` in the config.

RadioBurstSpectraMetrics' regression target is ``target["spectra"]``, shape ``(B, T, F)``.
"""

import torch
import torchmetrics as tm  # Lots of possible metrics in here https://lightning.ai/docs/torchmetrics/stable/all-metrics.html

# Shape contract: scalar predictions and targets are (B, 1). Every metric below flattens with
# reshape(-1) rather than squeeze(-1): squeeze is shape-dependent and collapses a
# batch of one to a 0-d scalar, which then fails to broadcast against a (1,) target.
class RadioBurstMetrics:
    def __init__(self, mode: str, peak_amp_weight: float = 1.0):
        """
        Initialize RadioBurstMetrics class.

        Args:
            mode (str): Mode to use for metric evaluation. One of "train_loss",
                        "val_loss", "train_metrics", or "val_metrics".
            peak_amp_weight (float): Weight applied to the masked peak_amp MSE term
                        when combining it with the burst-classification BCE term into a
                        single scalar loss.
        """
        self.mode = mode
        self.peak_amp_weight = peak_amp_weight

        # Cache torchmetrics instances once (instead of recreating each call)
        self._rrse = tm.RelativeSquaredError(squared=False)
        self._f1 = tm.F1Score(task="binary")

    def _ensure_device(self, preds: torch.Tensor) -> None:
        """Move torchmetrics modules to the same device as ``preds``, if needed."""
        if self._rrse.device != preds.device:
            self._rrse = self._rrse.to(preds.device)
        if self._f1.device != preds.device:
            self._f1 = self._f1.to(preds.device)
    
    def _burst_mask(self, burst_target: torch.Tensor) -> torch.Tensor:
        return burst_target.reshape(-1).bool()

    def _masked_values(self, pred, target, burst_target):
        """Flattened (pred, target) over burst rows and finite target cells only.

        Works for any per-sample shape: (B, 1) peak_amp, (B, T, F) spectra. Non-finite
        targets are missing measurements (e.g. gaps in a spectrogram) and are skipped
        rather than filled.
        """
        mask = self._burst_mask(burst_target)
        pred, target = pred[mask], target[mask]
        finite = torch.isfinite(target)
        return pred[finite], target[finite]

    def _masked_mse(self, pred, target, burst_target):
        pred, target = self._masked_values(pred, target, burst_target)
        if target.numel() == 0:
            return torch.zeros((), device=pred.device)
        return torch.nn.functional.mse_loss(pred, target)

    def _masked_rrse(self, pred, target, burst_target):
        pred, target = self._masked_values(pred, target, burst_target)
        if target.numel() == 0:
            return torch.zeros((), device=pred.device)
        # _rrse is a stateful, single-output torchmetrics instance, so it is fed one
        # flattened stream, matching _masked_mse's all-elements reduction.
        return self._rrse(pred, target)


    def train_loss(
        self, preds: dict, target: dict
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate loss metrics for training.

        Args:
            preds (dict): Model predictions, with keys "burst_prob" (B, 1) and
                        "peak_amp" (B, 1).
            target (dict): Ground truth, with keys "burst" (B,) — the 0/1 burst label —
                        and "diagnostics" (B, 1) — the peak_amp target, NaN where
                        "burst" == 0.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: "bce" (burst classification) and
                                        "mse_peak_amp" (peak_amp regression, masked to
                                        burst rows).
                - list[float]: Weights for each loss, aligned with the dict's key order.
        """

        output_metrics = {}
        output_weights = []

        output_metrics["bce"] = torch.nn.functional.binary_cross_entropy(
            preds["burst_prob"].reshape(-1), target["burst"].reshape(-1).float()
        )
        output_weights.append(1.0)

        output_metrics["mse_peak_amp"] = self._masked_mse(
            preds["peak_amp"], target["diagnostics"], target["burst"]
        )
        output_weights.append(self.peak_amp_weight)

        return output_metrics, output_weights

    def val_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate the validation loss — the quantity logged as ``val_loss`` and used by
        ModelCheckpoint to select the best model.

        By default this delegates to ``train_loss``, so the monitored quantity has the same
        form as the training objective and the two cannot drift apart by accident. This is
        the hook to override when your task needs a different validation objective (a
        different weighting, a metric that is meaningful only on held-out data, etc.).

        Note that this is deliberately separate from ``val_metrics``: those are reported
        for information only and do not affect checkpoint selection.

        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated loss metrics.
                - list[float]: List of weights for each calculated metric.
        """
        return self.train_loss(preds, target)

    def train_metrics(
        self, preds: dict, target: dict
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate evaluation metrics for training.
        IMPORTANT:  These metrics are only for reporting purposes and do not
                    contribute to the training loss. Use only if you want to
                    monitor additional metrics during training.

        Args:
            preds (dict): Model predictions.
            target (dict): Ground truth.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated evaluation metrics.
                                        Keys are metric names, and values are the corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """
        output_metrics = {}
        output_weights = []

        self._ensure_device(preds["burst_prob"])
        output_metrics["f1"] = self._f1(preds["burst_prob"].reshape(-1), target["burst"].reshape(-1).int())
        output_weights.append(1)

        output_metrics["rrse"] = self._masked_rrse(preds["peak_amp"], target["diagnostics"], target["burst"])
        output_weights.append(1)


        return output_metrics, output_weights

    def val_metrics(
        self, preds: dict, target: dict
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """
        Calculate metrics for validation.

        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: Dictionary containing the calculated metrics.
                                        Keys are metric names (e.g., "mse"), and values are the
                                        corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """

        output_metrics = {}
        output_weights = []

        self._ensure_device(preds["burst_prob"])
        output_metrics["f1"] = self._f1(preds["burst_prob"].reshape(-1), target["burst"].reshape(-1).int())
        output_weights.append(1)

        output_metrics["mse"] = self._masked_mse(preds["peak_amp"], target["diagnostics"], target["burst"])
        output_weights.append(1)

        output_metrics["rrse"] = self._masked_rrse(preds["peak_amp"], target["diagnostics"], target["burst"])
        output_weights.append(1)

        return output_metrics, output_weights

    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Evaluate metrics for the mode set at construction time.

        Args:
            preds: Model output tensor. Shape depends on the application.
            target: Ground truth tensor to compare against.

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - Metric dictionary. Keys become logger metric names; values are
                  scalar tensors aggregated over the batch.
                - List of per-metric weights (used by RadioBurstLightningModule to
                  combine multiple loss terms into a single scalar).
        """

        match self.mode.lower():

            case "train_loss":
                return self.train_loss(preds, target)

            # No torch.no_grad() here, matching "train_loss": Lightning already disables
            # gradients during validation, so wrapping it would differ gratuitously from
            # the loss case this mirrors.
            case "val_loss":
                return self.val_loss(preds, target)

            case "train_metrics":
                with torch.no_grad():
                    return self.train_metrics(preds, target)

            case "val_metrics":
                with torch.no_grad():
                    return self.val_metrics(preds, target)

            case _:
                raise NotImplementedError(
                    f"{self.mode} is not implemented as a valid metric case."
                )


class RadioBurstSpectraMetrics(RadioBurstMetrics):
    """Metrics for HelioSpectformerBurst: a burst logit plus a full (T, F) spectrogram.

    Predictions: ``preds["burst_logit"]`` (B, 1) and ``preds["spectra"]`` (B, T, F).
    Targets: ``target["burst"]`` (B,) and ``target["spectra"]`` (B, T, F).

    BCE is computed on the logit with ``binary_cross_entropy_with_logits``: plain
    ``binary_cross_entropy`` on sigmoid outputs raises under CUDA autocast (bf16-mixed).
    """

    def __init__(self, mode: str, spectra_weight: float = 1.0):
        """
        Args:
            mode (str): One of "train_loss", "val_loss", "train_metrics", or "val_metrics".
            spectra_weight (float): Weight on the masked spectrogram MSE relative to the
                        burst-classification BCE in the combined loss.
        """
        super().__init__(mode)
        self.spectra_weight = spectra_weight

    def _f1_from_logit(self, preds: dict, target: dict) -> torch.Tensor:
        self._ensure_device(preds["burst_logit"])
        burst_prob = torch.sigmoid(preds["burst_logit"]).reshape(-1)
        return self._f1(burst_prob, target["burst"].reshape(-1).int())

    def train_loss(
        self, preds: dict, target: dict
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """BCE on the burst logit, plus spectrogram MSE masked to burst rows."""
        output_metrics = {}
        output_weights = []

        output_metrics["bce"] = torch.nn.functional.binary_cross_entropy_with_logits(
            preds["burst_logit"].reshape(-1), target["burst"].reshape(-1).float()
        )
        output_weights.append(1.0)

        output_metrics["mse_spectra"] = self._masked_mse(
            preds["spectra"], target["spectra"], target["burst"]
        )
        output_weights.append(self.spectra_weight)

        return output_metrics, output_weights

    def train_metrics(
        self, preds: dict, target: dict
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Reported only: burst F1 and masked spectrogram RRSE."""
        output_metrics = {"f1": self._f1_from_logit(preds, target)}
        output_metrics["rrse_spectra"] = self._masked_rrse(
            preds["spectra"], target["spectra"], target["burst"]
        )
        return output_metrics, [1, 1]

    def val_metrics(
        self, preds: dict, target: dict
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Reported only: burst F1, masked spectrogram MSE and RRSE."""
        output_metrics = {"f1": self._f1_from_logit(preds, target)}
        output_metrics["mse_spectra"] = self._masked_mse(
            preds["spectra"], target["spectra"], target["burst"]
        )
        output_metrics["rrse_spectra"] = self._masked_rrse(
            preds["spectra"], target["spectra"], target["burst"]
        )
        return output_metrics, [1, 1, 1]
