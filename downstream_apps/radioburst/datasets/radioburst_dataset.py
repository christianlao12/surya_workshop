"""
Dataset for 24 h Type II radio-burst forecasting.

Each sample is one Surya frame at an issue time t (00/06/12/18 UTC, see prepare_data.py)
and two yes/no labels:

    type2     — a Type II burst starts in (t, t + horizon]
    type2_ip  — an interplanetary-reaching Type II (ends below 1 MHz) starts in that window

The labels are nested: every type2_ip positive is also a type2 positive. Their exact
definition lives in ``labels.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from downstream_apps.radioburst.labels import make_labels, onset_in_window
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset


class TypeIIDataset(HelioNetCDFDataset):
    """HelioNetCDFDataset plus the two Type II labels.

    The index CSV already holds exactly the issue times (``prepare_data.py`` writes one
    frame every 6 h), so the issue time *is* the frame time: no timestamp matching, and the
    input can never come from after the issue time. Only the frame at offset 0 is loaded.

    Args:
        ds_events_path: CSV with one row per onset (``onset``, ``reaches_ip``, ``behind_limb``).
        ds_horizon: Label window length, e.g. ``"24h"``.
        ds_negative_ratio: Negatives kept per positive (``type2``), applied only when
            ``phase == "train"``. Validation and test keep the natural event rate so their
            scores stay honest. ``None`` keeps every sample.
        ds_exclude_behind_limb: Drop samples whose window contains a behind-the-limb onset.
            They are not relabelled as negatives: the Sun really did produce a Type II.
        subsample_seed: Seed for the negative subsampling, so the kept set is reproducible.
        max_number_of_samples: Cap the dataset length (quick experiments).
        return_surya_stack: If False, skip loading SDO data and return labels only.
        **kwargs: Every ``HelioNetCDFDataset`` argument (``index_path``, ``scalers``, ...).
    """

    def __init__(
        self,
        ds_events_path: str,
        ds_horizon: str = "24h",
        ds_negative_ratio: float | None = None,
        ds_exclude_behind_limb: bool = False,
        subsample_seed: int = 0,
        max_number_of_samples: int | None = None,
        return_surya_stack: bool = True,
        **kwargs,
    ):
        # The labels come from the event list, so future Surya frames are never needed.
        kwargs.setdefault("load_forecast_frames", False)
        super().__init__(**kwargs)
        self.return_surya_stack = return_surya_stack

        events = pd.read_csv(ds_events_path, parse_dates=["onset"])
        issue = pd.DatetimeIndex(self.valid_indices)
        labels = make_labels(events, issue, ds_horizon)

        if ds_exclude_behind_limb:
            hidden = onset_in_window(events.loc[events.behind_limb == 1, "onset"], issue, ds_horizon)
            labels = labels[~hidden]

        if ds_negative_ratio is not None and self.phase == "train":
            labels = self._subsample_negatives(labels, ds_negative_ratio, subsample_seed)

        if max_number_of_samples is not None:
            labels = labels.iloc[:max_number_of_samples]

        self.labels = labels
        self.valid_indices = list(labels.index)
        self.adjusted_length = len(labels)

    @staticmethod
    def _subsample_negatives(labels: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
        """Keep every positive and ``ratio`` negatives per positive, chosen at random."""
        positive = labels["type2"] == 1
        negatives = labels.index[~positive]
        n_keep = min(len(negatives), int(round(ratio * positive.sum())))
        kept = np.random.default_rng(seed).choice(negatives, size=n_keep, replace=False)
        return labels[positive | labels.index.isin(kept)]

    def __len__(self) -> int:
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with:
            labels (np.ndarray[float32], shape (2,)): ``[type2, type2_ip]``, in
                ``labels.LABELS`` order.
            ds_index (str): the issue time, ISO format.
        plus, when ``return_surya_stack`` is True, every key from
        ``HelioNetCDFDataset.__getitem__`` (``ts``, ``time_delta_input``, ...).
        """
        sample = super().__getitem__(idx=idx) if self.return_surya_stack else {}
        sample["labels"] = self.labels.iloc[idx].to_numpy(dtype=np.float32)
        sample["ds_index"] = self.labels.index[idx].isoformat()
        return sample
