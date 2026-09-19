"""
Metrics for the radio-burst downstream app.

Two classes, one per model:
- RadioBurstMetrics         — for the linear baseline (TwoStageBurstModel): burst probability
                              plus a scalar ``peak_amp``.
- RadioBurstSpectraMetrics  — for the Surya fine-tuning head (HelioSpectformerBurst): burst
                              logit plus a full (T, F) spectrogram.

Both define four metric sets:
- "train_loss"    — differentiable loss that drives backpropagation (burst BCE + masked,
                    variance-normalized MSE).
- "val_loss"      — the quantity logged as `val_loss` and used to select checkpoints.
                    Defaults to the same loss as "train_loss"; override it when your task
                    needs a different validation objective.
- "train_metrics" — non-differentiable metrics logged during training (F1 + RRSE).
- "val_metrics"   — metrics logged at validation for reporting only (F1 + RRSE).
                    These do NOT influence checkpoint selection — "val_loss" does.
                    The masked MSE is deliberately absent: "val_loss" already reports it
                    under the same name, and logging it twice produced two identical
                    curves.

**Loss weights.** The terms are returned as separate dict entries plus a list of weights;
``RadioBurstLightningModule._combine_losses()`` forms the weighted sum. Both weights are
constructor arguments — ``burst_weight`` alongside ``peak_amp_weight`` or ``spectra_weight`` —
and default to 1.0, so the defaults reproduce the unweighted objective exactly. They are also
exposed as ``loss_weights`` so the training module can record them with the run: the weights
live in no config file, so nothing else would tell two runs apart afterwards.

Component losses are logged **unweighted** while "train_loss"/"val_loss" are the weighted
sums, so at any weight other than 1.0 the components do not add up to the total. That is
deliberate: a raw component curve stays comparable across runs that used different weights.

Raising ``burst_weight`` relative to the regression weight shifts capacity in the shared
embedding towards classification. Comparable loss *values* do not imply comparable influence
there — the BCE is a mean over ``B`` numbers while the spectrogram MSE is a mean over
``B x T x F`` (~69,000 at the shipped shape), so the two gradients arriving at the pooled
embedding can differ in norm by orders of magnitude at equal loss value.

**Why the regression term is normalized (``_masked_nmse``, not ``_masked_mse``).** The two
regression targets live at very different scales: ``target["spectra"]``'s cells are
median/MAD-standardized to roughly unit variance by ``SpectraTemplateNormalizer``, but
``peak_amp`` is a ``nanmax`` over that same field — deliberately *not* compressed by that
normalization (see its docstring), so its own variance can be orders of magnitude larger
(observed ~7,000–8,500 early in training, against a ``burst_bce`` of ~0.6–0.8). A fixed
``peak_amp_weight``/``spectra_weight`` would have to cancel that scale out and then be
re-tuned as the term shrinks over training. Dividing the masked MSE by variance
(``_masked_nmse``) fixes the scale at (roughly) 1 regardless of the target's units, so the
weight only expresses *how much this task should matter* relative to ``burst_bce``.

**``spectra`` estimates that variance per batch; ``peak_amp`` cannot.** ``spectra``'s
masked target is ``B x T x F`` cells — generally thousands even at ``batch_size: 2`` — so
a per-batch variance is already a stable estimate and this is a no-op (its cells are
already near unit variance). ``peak_amp``'s masked target is *one scalar per burst row*,
so at ``batch_size: 2`` and a typical catalog's ~44% burst rate, most non-empty batches
carry exactly one burst row: too few for `.var()` to be anything but 0, which silently
falls back to raw (thousands-scale) MSE almost every step — not an occasional edge case,
the common one. ``RadioBurstMetrics`` therefore takes an optional ``peak_amp_scale``
constructor argument: a fixed variance, fit once over every burst row in the training
catalog by ``spectra_transform.compute_peak_amp_scale``, that replaces the per-batch
estimate entirely. Pass it; the per-batch fallback (``peak_amp_scale=None``) is not
enough on its own at this batch size.

The __call__ method selects the appropriate metric set based on the mode passed at
construction time. The dictionary keys returned by each method become the metric names
propagated to the logger (e.g. WandB, CSV).

The regression terms are masked to burst rows (``target["burst"] == 1``): quiet windows
only train the classifier. Non-finite target cells (missing measurements) are skipped too.
A batch with no burst rows at all makes that term exactly 0.0 and contributes no gradient to
it, which also drags down its epoch average — ``RadioBurstLightningModule`` logs
``train_burst_rows``/``val_burst_rows`` so you can see how often that happens.

RadioBurstMetrics' regression target is a single scalar, standardized ``peak_amp`` — the
amplitude TwoStageBurstModel regresses, in ``SpectraTemplateNormalizer``'s standardized
space (``downstream_apps/radioburst/spectra_transform.py``), to reconstruct a full
spectrogram (``downstream_apps/radioburst/models/simple_baseline.py``). The ground truth
for it is *derived*, not read off the catalog: ``_standardized_peak_amp_target()`` takes
the ``nanmax`` over ``target["spectra"]``'s last two dims. This is deliberately not the
catalog's raw ``peak_amp`` column (still available as ``target["diagnostics"]``, per
``ds_diagnostics_columns: [peak_amp]`` in the config, but unused by training now) —
deriving it from the actual standardized spectrogram is robust to real bursts deviating
from the idealized median-template shape, and it makes this class's target and
RadioBurstSpectraMetrics' target the same quantity, in the same space.

RadioBurstSpectraMetrics' regression target is ``target["spectra"]``, shape ``(B, T, F)``.
"""

import torch
import torchmetrics as tm  # Lots of possible metrics in here https://lightning.ai/docs/torchmetrics/stable/all-metrics.html

# Shape contract: scalar predictions and targets are (B, 1). Every metric below flattens with
# reshape(-1) rather than squeeze(-1): squeeze is shape-dependent and collapses a
# batch of one to a 0-d scalar, which then fails to broadcast against a (1,) target.
class RadioBurstMetrics:
    def __init__(
        self,
        mode: str,
        burst_weight: float = 1.0,
        peak_amp_weight: float = 1.0,
        peak_amp_scale: float | None = None,
    ):
        """
        Initialize RadioBurstMetrics class.

        Args:
            mode (str): Mode to use for metric evaluation. One of "train_loss",
                        "val_loss", "train_metrics", or "val_metrics".
            burst_weight (float): Weight applied to the burst-classification BCE term
                        when combining the loss terms into a single scalar.
            peak_amp_weight (float): Weight applied to the masked peak_amp MSE term
                        when combining it with the burst-classification BCE term into a
                        single scalar loss.
            peak_amp_scale (float | None): Fixed variance to normalize the masked
                        peak_amp MSE by, instead of estimating it from each batch. See
                        ``_masked_nmse`` for why a per-batch estimate is unreliable for
                        this particular target, and
                        ``spectra_transform.compute_peak_amp_scale`` for how to fit this
                        once, from the training catalog. ``None`` (the default) falls
                        back to the per-batch estimate.

        Both weights default to 1.0, which reproduces the unweighted sum. Only their ratio
        matters for the direction of the gradient; their absolute scale interacts with the
        learning rate, so prefer holding burst_weight at 1.0 and moving the regression
        weight.
        """
        self.mode = mode
        self.burst_weight = burst_weight
        self.peak_amp_weight = peak_amp_weight
        self.peak_amp_scale = peak_amp_scale

        # Cache torchmetrics instances once (instead of recreating each call)
        self._rrse = tm.RelativeSquaredError(squared=False)
        self._f1 = tm.F1Score(task="binary")

    @property
    def loss_weights(self) -> dict[str, float]:
        """The weights this instance applies, for the training module to log with the run.

        Nothing else records them: they are constructor arguments, absent from the config
        YAML, so without this a finished run gives no way to tell which weighting produced
        it. ``RadioBurstLightningModule`` reads this via ``getattr`` and hands it to
        ``save_hyperparameters``.
        """
        weights = {"burst_weight": self.burst_weight, "peak_amp_weight": self.peak_amp_weight}
        if self.peak_amp_scale is not None:
            weights["peak_amp_scale"] = float(self.peak_amp_scale)
        return weights

    def _ensure_device(self, preds: torch.Tensor) -> None:
        """Move torchmetrics modules to the same device as ``preds``, if needed."""
        if self._rrse.device != preds.device:
            self._rrse = self._rrse.to(preds.device)
        if self._f1.device != preds.device:
            self._f1 = self._f1.to(preds.device)
    
    def _burst_mask(self, burst_target: torch.Tensor) -> torch.Tensor:
        return burst_target.reshape(-1).bool()

    @staticmethod
    def _nanmax_over_last_dims(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
        """NaN-aware max over ``dims``; NaN wherever every value along them is NaN.

        ``torch`` has no ``nanmax`` reduction, unlike ``numpy``: NaNs are filled with
        ``-inf`` before the max so they never win, then any position where *every* value
        was NaN (an all-gap row) is restored to NaN rather than left at ``-inf``.
        """
        is_nan = torch.isnan(x)
        filled = x.masked_fill(is_nan, float("-inf"))
        result = filled.amax(dim=dims)
        return result.masked_fill(is_nan.all(dim=dims), float("nan"))

    def _standardized_peak_amp_target(self, target: dict) -> torch.Tensor:
        """The standardized-space peak amplitude ground truth, shape ``(B, 1)``.

        Derived as the ``nanmax`` over ``target["spectra"]``'s ``(T, F)`` dims, rather
        than read from the catalog's raw ``peak_amp`` column — see the module docstring.
        """
        return self._nanmax_over_last_dims(target["spectra"], dims=(-2, -1)).unsqueeze(-1)

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

    def _masked_nmse(self, pred, target, burst_target, scale: float | None = None):
        """Masked MSE normalized by variance, so the result sits at (roughly) unit scale
        regardless of the target's raw units — see the module docstring for why
        ``peak_amp`` needs this and ``spectra`` mostly doesn't.

        Args:
            scale: A fixed variance to divide by, e.g. from
                ``spectra_transform.compute_peak_amp_scale``. When ``None`` (the
                default), the variance is instead estimated from this batch's own masked
                target — ``target.var()``.

                The per-batch estimate is what ``spectra`` should use: its masked target
                is ``B x T x F`` cells, generally thousands even at ``batch_size: 2``, so
                the per-batch variance is already a stable estimate of the population's.

                ``peak_amp`` needs a fixed ``scale`` instead: its masked target is *one
                scalar per burst row*, so at ``batch_size: 2`` and a typical catalog's
                ~44% burst rate, most non-empty batches carry exactly one burst row —
                too few for `.var()` to be anything but 0. The clamp below stops that
                from *blowing up* the loss (0 variance would divide by ~0), but it does
                not fix the underlying problem: it just silently falls back to raw MSE,
                which is still in the thousands. Only a scale fit across many burst rows
                (i.e. not per-batch) actually normalizes it.

        Either way the variance is floored at 1.0 rather than a small epsilon: an epsilon
        floor would turn a near-zero variance into an arbitrarily large ratio, whereas
        flooring at 1.0 makes the worst case degrade to plain MSE.
        """
        pred, target = self._masked_values(pred, target, burst_target)
        if target.numel() == 0:
            return torch.zeros((), device=pred.device)
        if scale is None:
            scale = target.var(unbiased=False)
        variance = torch.as_tensor(scale, dtype=pred.dtype, device=pred.device).clamp_min(1.0)
        return torch.nn.functional.mse_loss(pred, target) / variance

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
                        "peak_amp" (B, 1), standardized-space.
            target (dict): Ground truth, with keys "burst" (B,) — the 0/1 burst label —
                        and "spectra" (B, T, F), standardized-space, from which the
                        peak_amp target is derived (see
                        ``_standardized_peak_amp_target``).

        Returns:
            tuple[dict[str, torch.Tensor], list[float]]:
                - dict[str, torch.Tensor]: "burst_bce" (burst classification) and
                                        "peak_amp_nmse" (peak_amp regression, masked to
                                        burst rows and normalized by the masked target's
                                        variance — see ``_masked_nmse``). Both unweighted
                                        — the weights are returned alongside, not folded
                                        in.
                - list[float]: Weights for each loss, aligned with the dict's key order.
        """

        output_metrics = {}
        output_weights = []

        output_metrics["burst_bce"] = torch.nn.functional.binary_cross_entropy(
            preds["burst_prob"].reshape(-1), target["burst"].reshape(-1).float()
        )
        output_weights.append(self.burst_weight)

        output_metrics["peak_amp_nmse"] = self._masked_nmse(
            preds["peak_amp"],
            self._standardized_peak_amp_target(target),
            target["burst"],
            scale=self.peak_amp_scale,
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
        output_metrics["burst_f1"] = self._f1(preds["burst_prob"].reshape(-1), target["burst"].reshape(-1).int())
        output_weights.append(1)

        output_metrics["peak_amp_rrse"] = self._masked_rrse(
            preds["peak_amp"], self._standardized_peak_amp_target(target), target["burst"]
        )
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
                                        Keys are metric names (e.g., "burst_f1"), and values are the
                                        corresponding torch.Tensor values.
                - list[float]: List of weights for each calculated metric.
        """

        output_metrics = {}
        output_weights = []

        self._ensure_device(preds["burst_prob"])
        output_metrics["burst_f1"] = self._f1(preds["burst_prob"].reshape(-1), target["burst"].reshape(-1).int())
        output_weights.append(1)

        # No masked MSE here: val_loss already reports it as "peak_amp_nmse". Computing it
        # in both places logged two bit-identical curves.
        output_metrics["peak_amp_rrse"] = self._masked_rrse(
            preds["peak_amp"], self._standardized_peak_amp_target(target), target["burst"]
        )
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

    def __init__(self, mode: str, burst_weight: float = 1.0, spectra_weight: float = 1.0):
        """
        Args:
            mode (str): One of "train_loss", "val_loss", "train_metrics", or "val_metrics".
            burst_weight (float): Weight on the burst-classification BCE in the combined loss.
            spectra_weight (float): Weight on the masked spectrogram MSE relative to the
                        burst-classification BCE in the combined loss.

        Both default to 1.0, reproducing the unweighted sum. Turn spectra_weight down to give
        the classifier more of the shared embedding; the per-term curves
        (train_loss_burst_bce vs train_loss_spectra_nmse, and val_metric_spectra_rrse) are what
        tell you whether it worked, not the combined val_loss.
        """
        super().__init__(mode, burst_weight=burst_weight)
        self.spectra_weight = spectra_weight

    @property
    def loss_weights(self) -> dict[str, float]:
        """See ``RadioBurstMetrics.loss_weights``; this model's regression term is the spectra."""
        return {"burst_weight": self.burst_weight, "spectra_weight": self.spectra_weight}

    def _f1_from_logit(self, preds: dict, target: dict) -> torch.Tensor:
        self._ensure_device(preds["burst_logit"])
        burst_prob = torch.sigmoid(preds["burst_logit"]).reshape(-1)
        return self._f1(burst_prob, target["burst"].reshape(-1).int())

    def train_loss(
        self, preds: dict, target: dict
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """BCE on the burst logit, plus variance-normalized spectrogram MSE masked to burst rows.

        Both terms are returned unweighted, with their weights alongside, so the caller logs
        the raw components and combines them into the scalar that is backpropagated. The
        regression term uses ``_masked_nmse`` rather than raw MSE (see the module
        docstring); for this target it is close to a no-op since the masked cells are
        already near unit variance by construction of ``SpectraTemplateNormalizer``.
        """
        output_metrics = {}
        output_weights = []

        output_metrics["burst_bce"] = torch.nn.functional.binary_cross_entropy_with_logits(
            preds["burst_logit"].reshape(-1), target["burst"].reshape(-1).float()
        )
        output_weights.append(self.burst_weight)

        output_metrics["spectra_nmse"] = self._masked_nmse(
            preds["spectra"], target["spectra"], target["burst"]
        )
        output_weights.append(self.spectra_weight)

        return output_metrics, output_weights

    def _spectra_peak_mse(self, preds: dict, target: dict) -> torch.Tensor:
        """Masked MSE between the standardized-space peak of predicted vs. target spectra.

        Both are reduced to a per-sample ``(B, 1)`` max over ``(T, F)`` first (the same
        derivation ``RadioBurstMetrics`` uses for the baseline's ``peak_amp`` target — see
        that class's ``_standardized_peak_amp_target``), then compared with the usual
        burst-row masking. This is reported only, not backpropagated: it answers whether
        the full-spectrogram decoder gets the standardized peak amplitude right, without
        adding a dedicated scalar head for it.

        Because ``SpectraTemplateNormalizer`` subtracts a per-cell ``log_template`` rather
        than a single global constant, the ``(t, f)`` location of this standardized-space
        maximum is not guaranteed to be the same cell as the raw-flux maximum — this
        metric is about the standardized peak, not a proxy for the catalog's raw
        ``peak_amp`` column.
        """
        pred_peak = self._nanmax_over_last_dims(preds["spectra"], dims=(-2, -1)).unsqueeze(-1)
        target_peak = self._standardized_peak_amp_target(target)
        return self._masked_mse(pred_peak, target_peak, target["burst"])

    def train_metrics(
        self, preds: dict, target: dict
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Reported only: burst F1, masked spectrogram RRSE, and standardized peak MSE."""
        output_metrics = {"burst_f1": self._f1_from_logit(preds, target)}
        output_metrics["spectra_rrse"] = self._masked_rrse(
            preds["spectra"], target["spectra"], target["burst"]
        )
        output_metrics["spectra_peak_mse"] = self._spectra_peak_mse(preds, target)
        return output_metrics, [1, 1, 1]

    def val_metrics(
        self, preds: dict, target: dict
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Reported only: burst F1, masked spectrogram RRSE, and standardized peak MSE.

        No masked full-spectrogram MSE: val_loss already reports it as "spectra_nmse". RRSE
        is the more readable of the two anyway — it is scaled by the variance of the
        target, so > 1.0 means the decoder is doing worse than predicting the target mean.
        """
        output_metrics = {"burst_f1": self._f1_from_logit(preds, target)}
        output_metrics["spectra_rrse"] = self._masked_rrse(
            preds["spectra"], target["spectra"], target["burst"]
        )
        output_metrics["spectra_peak_mse"] = self._spectra_peak_mse(preds, target)
        return output_metrics, [1, 1, 1]
