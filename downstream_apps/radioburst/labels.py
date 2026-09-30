"""
Label definitions for Type II forecasting, in one place.

Used by the dataset (training labels), ``prepare_data.py`` (split summaries) and
``evaluate.py`` (test labels and the persistence baseline), so all three agree on exactly
what "a Type II in the next 24 h" means. Pure pandas/numpy: no torch, so it is importable
and testable anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Order of the two outputs everywhere: labels, logits, metrics.
LABELS = ("type2", "type2_ip")


def onset_in_window(onsets, issue_times, horizon) -> np.ndarray:
    """True where at least one onset falls in (t, t + horizon] for each issue time t.

    The window is open at t: an onset exactly at the issue time is already happening, so
    it belongs to the past, not to the forecast. It is closed at t + horizon.
    """
    onsets = np.sort(pd.to_datetime(pd.Series(onsets)).to_numpy())
    t = pd.to_datetime(pd.Series(issue_times)).to_numpy()
    h = pd.Timedelta(horizon).to_timedelta64()
    return np.searchsorted(onsets, t + h, side="right") > np.searchsorted(onsets, t, side="right")


def onset_in_past(onsets, issue_times, horizon) -> np.ndarray:
    """True where at least one onset falls in (t - horizon, t]: the persistence baseline."""
    t = pd.to_datetime(pd.Series(issue_times))
    return onset_in_window(onsets, t - pd.Timedelta(horizon), horizon)


def make_labels(events: pd.DataFrame, issue_times, horizon) -> pd.DataFrame:
    """The two labels for each issue time, as float32 columns in ``LABELS`` order.

    ``events`` needs an ``onset`` (datetime) and a ``reaches_ip`` (0/1) column.
    """
    index = pd.DatetimeIndex(issue_times)
    return pd.DataFrame(
        {
            "type2": onset_in_window(events["onset"], index, horizon),
            "type2_ip": onset_in_window(events.loc[events["reaches_ip"] == 1, "onset"], index, horizon),
        },
        index=index,
    ).astype(np.float32)
