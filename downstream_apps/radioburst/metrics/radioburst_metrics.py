"""
Loss and validation scores for Type II forecasting.

Two pieces, because they work at different scales:

- ``TypeIIMetrics`` — the per-batch loss, in the template's four-mode pattern
  (``train_loss``, ``val_loss``, ``train_metrics``, ``val_metrics``).
- ``ValidationScores`` — ROC-AUC, PR-AUC and TSS over the *whole* validation epoch.

Why not per-batch metrics? At a ~5% event rate and a batch size of 2, most batches hold
no positive at all, so a per-batch AUC or TSS is undefined or meaningless, and averaging
those over an epoch is not the epoch's score. ``train_metrics``/``val_metrics`` therefore
return nothing here, and the Lightning module feeds every validation batch into
``ValidationScores`` and logs its result once per epoch.

**Class imbalance** is handled by subsampling negatives in the training set
(``data.ds_negative_ratio``), not by ``pos_weight`` in the loss. Every dropped row is
negative for both labels, so one exact correction of the predicted odds undoes it for both
outputs at evaluation time (see ``evaluate.py``). Stacking ``pos_weight`` on top would make
that correction two-step and label-specific.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.aggregation import SumMetric
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryROC

from downstream_apps.radioburst.labels import LABELS


class TypeIIMetrics:
    """Binary cross-entropy on the two logits, one loss term per label.

    Args:
        mode: ``"train_loss"``, ``"val_loss"``, ``"train_metrics"`` or ``"val_metrics"``.
        type2_weight: Weight of the ``type2`` BCE term in the combined loss.
        type2_ip_weight: Weight of the ``type2_ip`` BCE term.

    Every mode is called as ``metrics(logits, labels)`` with both of shape ``(B, 2)`` in
    ``LABELS`` order, and returns ``(dict_of_scalars, list_of_weights)``. The Lightning
    module forms the weighted sum; the components are logged unweighted.
    """

    def __init__(self, mode: str, type2_weight: float = 1.0, type2_ip_weight: float = 1.0):
        if mode not in ("train_loss", "val_loss", "train_metrics", "val_metrics"):
            raise ValueError(f"Unknown mode {mode!r}")
        self.mode = mode
        self.weights = {"type2": type2_weight, "type2_ip": type2_ip_weight}

    @property
    def loss_weights(self) -> dict[str, float]:
        """The weights, so the Lightning module can record them with the run."""
        return {f"{name}_weight": w for name, w in self.weights.items()}

    def train_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        # float(): under bf16 autocast the logits arrive in bf16; BCE is computed in fp32.
        bce = F.binary_cross_entropy_with_logits(
            logits.float(), labels.float(), reduction="none"
        ).mean(dim=0)  # (2,)
        losses = {f"bce_{name}": bce[k] for k, name in enumerate(LABELS)}
        return losses, [self.weights[name] for name in LABELS]

    def val_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        """Same form as the training loss, so the monitored quantity cannot drift from it.

        Note that validation keeps the natural event rate while training is subsampled, so
        the two values are not directly comparable — only val_loss across epochs is.
        """
        return self.train_loss(logits, labels)

    def train_metrics(self, logits, labels):
        return {}, []  # see the module docstring: scores are computed per epoch

    def val_metrics(self, logits, labels):
        return {}, []

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor):
        return getattr(self, self.mode)(logits, labels)


class ValidationScores(nn.Module):
    """Accumulates predictions over an epoch and scores them once, per label.

    Built from torchmetrics objects, so under DDP the predictions of every process are
    gathered before scoring.

    ``compute()`` returns, for each label in ``LABELS``:
        ``<label>_event_rate``  fraction of positives in the epoch
        ``<label>_roc_auc``     threshold-free ranking quality
        ``<label>_pr_auc``      average precision; its no-skill level is the event rate
        ``<label>_tss_max``     max over thresholds of TPR - FPR. **Optimistic**: the
                                threshold is chosen on the same data it is scored on. For
                                monitoring only; evaluate.py picks the threshold on val
                                and scores test with it.
    The last three are omitted for a label with no positives (or no negatives) in the
    epoch, where they are undefined.
    """

    def __init__(self):
        super().__init__()
        self.per_label = nn.ModuleDict({
            name: nn.ModuleDict({
                "roc_auc": BinaryAUROC(),
                "pr_auc": BinaryAveragePrecision(),
                "roc": BinaryROC(),
                "positives": SumMetric(),
                "count": SumMetric(),
            })
            for name in LABELS
        })

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        probs = torch.sigmoid(logits.detach().float())
        labels = labels.detach().long()
        for k, name in enumerate(LABELS):
            m = self.per_label[name]
            for key in ("roc_auc", "pr_auc", "roc"):
                m[key].update(probs[:, k], labels[:, k])
            m["positives"].update(labels[:, k].sum().float())
            m["count"].update(torch.tensor(float(labels.shape[0]), device=labels.device))

    def compute(self) -> dict[str, torch.Tensor]:
        scores = {}
        for name in LABELS:
            m = self.per_label[name]
            positives, count = m["positives"].compute(), m["count"].compute()
            scores[f"{name}_event_rate"] = positives / count
            if 0 < positives < count:
                scores[f"{name}_roc_auc"] = m["roc_auc"].compute()
                scores[f"{name}_pr_auc"] = m["pr_auc"].compute()
                fpr, tpr, _ = m["roc"].compute()
                scores[f"{name}_tss_max"] = (tpr - fpr).max()
        return scores

    def reset(self) -> None:
        for m in self.per_label.values():
            for metric in m.values():
                metric.reset()
