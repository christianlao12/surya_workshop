"""Tests for the label definitions in labels.py — what "a Type II in the next 24 h" means.

An off-by-one here silently shifts every label, so the window edges are pinned exactly.
"""

import numpy as np
import pandas as pd

from downstream_apps.radioburst.labels import LABELS, make_labels, onset_in_past, onset_in_window

T = pd.Timestamp("2015-03-01 06:00")


def test_onset_inside_the_window_is_positive():
    assert onset_in_window([T + pd.Timedelta("5h")], [T], "24h").tolist() == [True]


def test_onset_exactly_at_the_issue_time_is_not_a_forecast():
    """The window is open at t: a burst already starting at t is the past, not the future."""
    assert onset_in_window([T], [T], "24h").tolist() == [False]


def test_onset_exactly_at_the_horizon_is_included():
    assert onset_in_window([T + pd.Timedelta("24h")], [T], "24h").tolist() == [True]


def test_onset_just_after_the_horizon_is_excluded():
    assert onset_in_window([T + pd.Timedelta("24h") + pd.Timedelta("1min")], [T], "24h").tolist() == [False]


def test_onset_before_the_issue_time_is_excluded():
    assert onset_in_window([T - pd.Timedelta("1h")], [T], "24h").tolist() == [False]


def test_unsorted_onsets_and_many_issue_times():
    onsets = [T + pd.Timedelta("30h"), T + pd.Timedelta("2h")]  # deliberately unsorted
    issue = [T, T + pd.Timedelta("6h"), T + pd.Timedelta("12h")]
    # T: onset at +2h. T+6h: +30h is 24h later -> inside. T+12h: +30h is 18h later -> inside.
    assert onset_in_window(onsets, issue, "24h").tolist() == [True, True, True]
    issue_late = [T + pd.Timedelta("31h")]
    assert onset_in_window(onsets, issue_late, "24h").tolist() == [False]


def test_no_onsets_gives_all_negative():
    assert onset_in_window([], [T, T + pd.Timedelta("6h")], "24h").tolist() == [False, False]


def test_persistence_looks_back_including_the_issue_time():
    assert onset_in_past([T], [T], "24h").tolist() == [True]
    assert onset_in_past([T - pd.Timedelta("24h")], [T], "24h").tolist() == [False]
    assert onset_in_past([T - pd.Timedelta("23h")], [T], "24h").tolist() == [True]
    assert onset_in_past([T + pd.Timedelta("1h")], [T], "24h").tolist() == [False]


def test_make_labels_columns_order_and_nesting():
    events = pd.DataFrame({
        "onset": [T + pd.Timedelta("3h"), T + pd.Timedelta("40h")],
        "reaches_ip": [0, 1],
    })
    issue = pd.date_range(T, periods=4, freq="6h")  # T, +6, +12, +18
    labels = make_labels(events, issue, "24h")

    assert tuple(labels.columns) == LABELS
    assert labels.dtypes.unique().tolist() == [np.float32]
    # +3h onset (not IP) only covers the first issue time; the +40h IP onset is within 24 h
    # of +18h only.
    assert labels["type2"].tolist() == [1, 0, 0, 1]
    assert labels["type2_ip"].tolist() == [0, 0, 0, 1]
    # Nested: an IP positive is always a Type II positive.
    assert (labels["type2_ip"] <= labels["type2"]).all()
