"""Tests for prepare_data.py: event parsing and the train/val/test splits.

The split tests guard against leakage — a label window or a near-identical neighbouring
frame crossing a split boundary — which would inflate every score without any error.
"""

import pandas as pd
import pytest

from downstream_apps.radioburst import prepare_data as pdata

SPLITS_DIR = pdata.DATA_DIR / "splits"


# --------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------

def write_catalog(path, rows):
    header = ("start_date,start_time,end_date,end_time,freq_start_kHz,freq_end_kHz,flare_loc,"
              "flare_noaa_ar,flare_importance,cme_date,cme_time,cme_cpa_deg,cme_width_deg,"
              "cme_speed_kms,plots,remarks")
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def test_build_events_filters_to_the_sdo_era_and_parses_flags(tmp_path):
    catalog = write_catalog(tmp_path / "cat.csv", [
        # Pre-SDO row with a placeholder that would break int parsing if not filtered first.
        "1997/01/20,08:56,01/20,09:02,14000,????,------,-----,----,--/--,--:--,----,----,----,PHTX,",
        "2012/03/07,01:00,03/07,08:00,16000,150,N17E27,11429,X5.4,03/07,00:24,Halo,360,2684,PHTX,",
        "2011/02/15,02:10,02/15,07:00,16000,1500,S20W10,11158,X2.2,02/15,02:24,Halo,360,669,PHTX,",
        "2013/05/22,13:20,05/22,20:00,14000,500,N15W90b,-----,M5.0,05/22,13:25,Halo,360,1466,PHTX,",
    ])
    events = pdata.build_events(catalog)

    assert len(events) == 3                      # 1997 dropped
    assert events["onset"].is_monotonic_increasing
    by_year = events.set_index(events.onset.dt.year)
    assert by_year.loc[2012, "reaches_ip"] == 1  # ends at 150 kHz < 1 MHz
    assert by_year.loc[2011, "reaches_ip"] == 0  # ends at 1500 kHz
    assert by_year.loc[2013, "behind_limb"] == 1  # "...90b"
    assert by_year.loc[2012, "behind_limb"] == 0


# --------------------------------------------------------------------------------------
# Split assignment
# --------------------------------------------------------------------------------------

def test_cycle_25_is_test_and_nothing_before_it_is():
    t = pd.DatetimeIndex(["2019-12-31 18:00", "2020-01-01 00:00", "2024-06-01"])
    assert pdata.assign_split(t).tolist()[1:] == ["test", "test"]
    assert pdata.assign_split(t)[0] != "test"


def test_every_fifth_rotation_is_validation():
    starts = pdata.START + pdata.ROTATION * pd.Index(range(10)) + pd.Timedelta("1D")
    assert pdata.assign_split(pd.DatetimeIndex(starts)).tolist() == (["train"] * 4 + ["val"]) * 2


# --------------------------------------------------------------------------------------
# build_splits on a synthetic index
# --------------------------------------------------------------------------------------

@pytest.fixture
def splits(tmp_path):
    t = pd.date_range("2019-06-01", "2020-01-10", freq="12min")  # long enough to hold a val rotation
    index = pd.DataFrame({"path": [f"s3://b/{i}.nc" for i in range(len(t))], "timestep": t, "present": 1})
    index.to_csv(tmp_path / "full.csv")  # the real full index has an unnamed index column too
    return pdata.build_splits(tmp_path / "full.csv")


def test_each_split_keeps_only_its_own_issue_time_grid(splits):
    for name, df in splits.items():
        kept = pd.to_datetime(df["timestep"])
        assert len(kept) > 0, name
        assert ((kept.dt.minute == 0) & (kept.dt.hour % pdata.CADENCE_H[name] == 0)).all(), name


def test_missing_frames_are_dropped_not_replaced(tmp_path):
    t = pd.date_range("2015-01-01", "2015-03-01", freq="12min")
    index = pd.DataFrame({"path": "s3://b/x.nc", "timestep": t, "present": 1})
    missing = pd.Timestamp("2015-02-01 00:00")  # on every split's grid
    index.loc[index.timestep == missing, "present"] = 0
    index.to_csv(tmp_path / "full.csv")
    kept = pd.to_datetime(pd.concat(pdata.build_splits(tmp_path / "full.csv").values())["timestep"])
    assert missing not in set(kept)
    assert missing + pd.Timedelta("12min") not in set(kept)  # no neighbour stands in for it


def test_output_columns_match_what_the_dataset_reads(splits):
    for df in splits.values():
        assert list(df.columns) == ["path", "timestep", "present"]


def test_the_gap_keeps_samples_away_from_the_cycle_25_boundary(splits):
    train_val = pd.to_datetime(pd.concat([splits["train"], splits["val"]])["timestep"])
    test = pd.to_datetime(splits["test"]["timestep"])
    assert train_val.max() < pdata.TEST_START - pdata.GAP + pd.Timedelta("6h")
    assert test.min() >= pdata.TEST_START + pdata.GAP


# --------------------------------------------------------------------------------------
# The committed split files
# --------------------------------------------------------------------------------------

def load_committed():
    if not (SPLITS_DIR / "train.csv").exists():
        pytest.skip("run `python -m downstream_apps.radioburst.prepare_data` first")
    return {n: pd.to_datetime(pd.read_csv(SPLITS_DIR / f"{n}.csv")["timestep"]) for n in ("train", "val", "test")}


def test_committed_splits_are_disjoint_and_separated_by_the_gap():
    splits = load_committed()
    labelled = pd.concat([pd.Series(n, index=t) for n, t in splits.items()]).sort_index()
    assert labelled.index.is_unique
    # Wherever consecutive samples change split, they must be at least 2 * GAP apart.
    changes = labelled != labelled.shift()
    gaps = labelled.index.to_series().diff()[changes.to_numpy()].iloc[1:]
    assert (gaps >= 2 * pdata.GAP).all()


def test_committed_test_split_is_cycle_25_only():
    splits = load_committed()
    assert splits["test"].min() >= pdata.TEST_START
    assert splits["train"].max() < pdata.TEST_START and splits["val"].max() < pdata.TEST_START
