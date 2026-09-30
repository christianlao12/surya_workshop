"""Tests for TypeIIDataset: labels, subsampling and exclusions, without loading any SDO data.

``return_surya_stack=False`` skips the NetCDF reads, so only the index and event files are
needed. The frames are never opened.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from downstream_apps.radioburst.datasets.radioburst_dataset import TypeIIDataset

T0 = pd.Timestamp("2015-03-01 00:00")
ISSUE = pd.date_range(T0, periods=40, freq="6h")  # 10 days of issue times
CHANNELS = ["aia94", "hmi_m"]


def make_files(tmp_path, onsets, reaches_ip=None, behind_limb=None):
    index = tmp_path / "index.csv"
    pd.DataFrame({"path": [f"s3://b/{i}.nc" for i in range(len(ISSUE))], "timestep": ISSUE, "present": 1}) \
        .to_csv(index, index=False)
    n = len(onsets)
    events = tmp_path / "events.csv"
    pd.DataFrame({
        "onset": onsets,
        "reaches_ip": reaches_ip if reaches_ip is not None else [0] * n,
        "behind_limb": behind_limb if behind_limb is not None else [0] * n,
    }).to_csv(events, index=False)
    return index, events


def make_dataset(index, events, phase="train", **kwargs):
    scaler = SimpleNamespace(mean=0.0, std=1.0, epsilon=1.0, sl_scale_factor=1.0)
    return TypeIIDataset(
        ds_events_path=str(events),
        return_surya_stack=False,
        index_path=str(index),
        time_delta_input_minutes=[0],
        time_delta_target_minutes=60,
        n_input_timestamps=1,
        rollout_steps=0,
        scalers={ch: scaler for ch in CHANNELS},
        channels=CHANNELS,
        phase=phase,
        s3_cache_dir=str(index.parent),
        **kwargs,
    )


def test_labels_follow_the_window_definition(tmp_path):
    onset = T0 + pd.Timedelta("50h")  # IP-reaching
    ds = make_dataset(*make_files(tmp_path, [onset], reaches_ip=[1]))
    expected = ((ISSUE < onset) & (ISSUE + pd.Timedelta("24h") >= onset)).astype(np.float32)
    assert ds.labels["type2"].tolist() == expected.tolist()
    assert ds.labels["type2_ip"].tolist() == expected.tolist()


def test_sample_contract(tmp_path):
    ds = make_dataset(*make_files(tmp_path, [T0 + pd.Timedelta("3h")]))
    sample = ds[0]
    assert sample["labels"].dtype == np.float32 and sample["labels"].shape == (2,)
    assert sample["labels"].tolist() == [1.0, 0.0]
    assert sample["ds_index"] == T0.isoformat()
    assert len(ds) == len(ISSUE)


def test_every_frame_is_the_issue_time_itself(tmp_path):
    """The input is the frame at t: never after it, never a neighbour."""
    ds = make_dataset(*make_files(tmp_path, [T0 + pd.Timedelta("3h")]))
    assert [pd.Timestamp(t) for t in ds.valid_indices] == list(ds.labels.index)
    assert set(ds.labels.index) <= set(ISSUE)


def test_negative_subsampling_keeps_all_positives_and_the_ratio(tmp_path):
    onsets = [T0 + pd.Timedelta("30h"), T0 + pd.Timedelta("150h")]
    files = make_files(tmp_path, onsets)
    full = make_dataset(*files)
    sub = make_dataset(*files, ds_negative_ratio=1, subsample_seed=0)

    n_pos = int(full.labels["type2"].sum())
    assert sub.labels["type2"].sum() == n_pos
    assert (sub.labels["type2"] == 0).sum() == n_pos
    # The same seed keeps the same negatives.
    again = make_dataset(*files, ds_negative_ratio=1, subsample_seed=0)
    assert sub.labels.index.equals(again.labels.index)
    # The kept fraction is recorded, for the probability correction at evaluation.
    n_neg = int((full.labels["type2"] == 0).sum())
    assert sub.negative_keep_fraction == pytest.approx(n_pos / n_neg)
    assert full.negative_keep_fraction == 1.0


def test_validation_is_never_subsampled(tmp_path):
    files = make_files(tmp_path, [T0 + pd.Timedelta("30h")])
    val = make_dataset(*files, phase="val", ds_negative_ratio=1)
    assert len(val) == len(ISSUE)


def test_behind_limb_windows_are_dropped_not_relabelled(tmp_path):
    hidden = T0 + pd.Timedelta("100h")
    files = make_files(tmp_path, [T0 + pd.Timedelta("30h"), hidden], behind_limb=[0, 1])
    ds = make_dataset(*files, ds_exclude_behind_limb=True)
    in_hidden_window = (ISSUE < hidden) & (ISSUE + pd.Timedelta("24h") >= hidden)
    assert set(ds.labels.index) == set(ISSUE[~in_hidden_window])
    assert ds.labels["type2"].sum() > 0  # the visible event is still a positive


def test_max_samples_caps_the_length(tmp_path):
    ds = make_dataset(*make_files(tmp_path, [T0 + pd.Timedelta("3h")]), max_number_of_samples=5)
    assert len(ds) == 5 and len(ds.valid_indices) == 5


@pytest.mark.parametrize("horizon", ["12h", "48h"])
def test_horizon_is_configurable(tmp_path, horizon):
    onset = T0 + pd.Timedelta("60h")
    ds = make_dataset(*make_files(tmp_path, [onset]), ds_horizon=horizon)
    expected = (ISSUE < onset) & (ISSUE + pd.Timedelta(horizon) >= onset)
    assert ds.labels["type2"].astype(bool).tolist() == expected.tolist()
