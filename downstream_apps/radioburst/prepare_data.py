#!/usr/bin/env python3
"""
Build the Type II forecasting data files from their raw sources.

Run from the repo root:

    python -m downstream_apps.radioburst.prepare_data                 # events + splits
    python -m downstream_apps.radioburst.prepare_data --fetch-flares  # also re-download flares

Everything lands in ``downstream_apps/radioburst/data/``:

    cdaw_typeii_catalog.csv  INPUT. The CDAW Wind/WAVES decameter-hectometric (DH) Type II
                             catalog, https://cdaw.gsfc.nasa.gov/CME_list/radio/waves_type2.html
    typeii_events.csv        One row per Type II onset in the SDO era. The dataset derives its
                             labels from this file.
    mx_flares.csv            M/X flare start times (SSW Latest Events, via HEK). Used only by
                             the "M/X flare in the last 24 h" baseline in evaluate.py.
    splits/{train,val,test}.csv
                             Surya index rows at the issue times: one frame every 12 h
                             (train, test) or 24 h (val); see CADENCE_H.

The splits
----------
- **test**: 2020-01-01 onward -- Solar Cycle 25, which Surya's pretraining excluded
  (Surya paper, arXiv:2508.14112, section 2.1).
- **val**: every 5th solar rotation (27.2753 d) of 2010-05-13 .. 2019-12-31.
- **train**: the other rotations of that period.

A sample is kept only if everything within ``GAP`` of it belongs to the same split. That
stops two leaks: a label window (t, t + 24 h] reaching into another split, and near-identical
neighbouring frames landing on both sides of a boundary.

The generic ``workshop_infrastructure/data/split_csv_index.py`` is not used: it splits by
day-of-year and drops 2012 and 2022, which fits neither the cycle-25 test set nor the need
for whole solar rotations.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from downstream_apps.radioburst.labels import make_labels

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
FULL_INDEX = APP_DIR.parents[1] / "data" / "indices" / "surya_aws_s3_full_index.csv"

START = pd.Timestamp("2010-05-13")    # SDO science data start
END = pd.Timestamp("2025-01-01")      # catalog assumed complete up to here
TEST_START = pd.Timestamp("2020-01-01")
ROTATION = pd.Timedelta(days=27.2753)  # synodic Carrington rotation
VAL_EVERY = 5                          # every 5th rotation of the train period is val
# Hours between issue times, per split (issue times fall on multiples of this, UTC). Sized
# for a ~1 TB frame cache: train and val frames are cached and re-read every epoch, test
# frames are read once (streamed by evaluate.py). Neighbouring issue times share most of
# their 24 h label window, so sparser val/test issue times lose little information.
CADENCE_H = {"train": 12, "val": 24, "test": 12}
HORIZON = pd.Timedelta(hours=24)       # label window; must match data.ds_horizon
GAP = pd.Timedelta(days=2)             # buffer around every split boundary
IP_THRESHOLD_KHZ = 1000                # ending below 1 MHz = shock reached interplanetary space


# --------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------

def build_events(catalog_csv: Path) -> pd.DataFrame:
    """One row per Type II onset in [START, END), sorted by onset.

    ``behind_limb`` marks source flares the catalog locates behind the limb (``...90b``),
    which SDO cannot see.
    """
    cat = pd.read_csv(catalog_csv, dtype=str)
    cat["onset"] = pd.to_datetime(cat["start_date"] + " " + cat["start_time"], format="%Y/%m/%d %H:%M")
    # Filter first: some pre-SDO rows hold placeholders such as "????" in numeric columns.
    cat = cat[(cat["onset"] >= START) & (cat["onset"] < END)]
    freq_end = cat["freq_end_kHz"].astype(int)
    events = pd.DataFrame({
        "onset": cat["onset"],
        "reaches_ip": (freq_end < IP_THRESHOLD_KHZ).astype(int),
        "behind_limb": cat["flare_loc"].fillna("").str.endswith("b").astype(int),
        "freq_start_kHz": cat["freq_start_kHz"].astype(int),
        "freq_end_kHz": freq_end,
        "flare_loc": cat["flare_loc"],
        "flare_importance": cat["flare_importance"],
        "cme_speed_kms": pd.to_numeric(cat["cme_speed_kms"], errors="coerce"),
    })
    return events.sort_values("onset").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------

def assign_split(t: pd.DatetimeIndex) -> np.ndarray:
    """'train', 'val' or 'test' for each timestamp (before any gap is applied)."""
    rotation = ((t - START) // ROTATION).to_numpy()
    split = np.where(rotation % VAL_EVERY == VAL_EVERY - 1, "val", "train")
    return np.where(t >= TEST_START, "test", split)


def build_splits(full_index_csv: Path) -> dict[str, pd.DataFrame]:
    """Surya index rows at the issue times, grouped by split, with the boundary gap applied.

    Issue times are the present frames on each split's ``CADENCE_H`` grid (on the hour, at
    hours divisible by the cadence). A missing frame drops that issue time rather than being
    replaced by a neighbour, so the input is never later than the issue time.
    """
    index = pd.read_csv(full_index_csv, usecols=["path", "timestep", "present"])
    index["timestep"] = pd.to_datetime(index["timestep"])
    t = index["timestep"]
    in_period = (t >= START) & (t + HORIZON <= END)  # the label window must be fully labelled
    index = index[(t.dt.minute == 0) & in_period & (index["present"] == 1)].reset_index(drop=True)

    t = pd.DatetimeIndex(index["timestep"])
    split = assign_split(t)
    # Blocks are much longer than 2 * GAP, so checking both edges covers the whole window.
    clean = (assign_split(t - GAP) == split) & (assign_split(t + GAP) == split)
    index["timestep"] = index["timestep"].dt.strftime("%Y-%m-%d %H:%M:%S")
    splits = {}
    for name, hours in CADENCE_H.items():
        on_grid = (t.hour % hours == 0)
        splits[name] = index[clean & (split == name) & on_grid]
    return splits


# --------------------------------------------------------------------------------------
# M/X flares (baseline only)
# --------------------------------------------------------------------------------------

def fetch_mx_flares(start: pd.Timestamp = START, end: pd.Timestamp = END) -> pd.DataFrame:
    """M- and X-class flares from SSW Latest Events (LMSAL), via the HEK API.

    SSW Latest Events is used rather than the SWPC list because the SWPC entries in HEK
    often carry placeholder (0, 0) coordinates.
    """
    rows = []
    years = pd.date_range(start.to_period("Y").start_time, end, freq="YS")
    for a, b in zip(years, list(years[1:]) + [end]):
        page = 1
        while True:
            params = {
                "cosec": 2, "cmd": "search", "type": "column", "event_type": "fl",
                "event_starttime": f"{max(a, start):%Y-%m-%dT%H:%M:%S}",
                "event_endtime": f"{b:%Y-%m-%dT%H:%M:%S}",
                "event_coordsys": "helioprojective", "x1": -5000, "x2": 5000, "y1": -5000, "y2": 5000,
                "param0": "FRM_NAME", "op0": "=", "value0": "SSW Latest Events",
                "param1": "FL_GOESCLS", "op1": ">=", "value1": "M",
                "result_limit": 500, "page": page,
                "return": "event_starttime,event_peaktime,fl_goescls,hgs_x,hgs_y",
            }
            url = "https://www.lmsal.com/hek/her?" + urllib.parse.urlencode(params)
            with urllib.request.urlopen(url, timeout=300) as response:
                result = json.load(response)
            rows += result["result"]
            if not result.get("overmax"):
                break
            page += 1
        print(f"  flares: {a.year} done ({len(rows)} so far)")

    flares = pd.DataFrame(rows).rename(columns={
        "event_starttime": "start_time", "event_peaktime": "peak_time", "fl_goescls": "goes_class"})
    flares = flares.drop_duplicates(["start_time", "goes_class", "hgs_x", "hgs_y"])
    flares = flares[flares["goes_class"].str[0].isin(["M", "X"])]
    return flares.sort_values("start_time").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def _summary(splits: dict[str, pd.DataFrame], events: pd.DataFrame) -> None:
    print(f"\n{'split':<6} {'issue times':>11} {'~TB':>5} {'P(TypeII 24h)':>14} {'P(IP 24h)':>10}")
    for name, df in splits.items():
        labels = make_labels(events, pd.to_datetime(df["timestep"]), HORIZON)
        print(f"{name:<6} {len(df):>11,} {len(df) / 1000:>5.1f} "
              f"{labels['type2'].mean():>14.3f} {labels['type2_ip'].mean():>10.3f}")
    print("(~TB assumes ~1 GB per SDO frame, before any negative subsampling)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch-flares", action="store_true",
                        help="Re-download data/mx_flares.csv from HEK (~1 min).")
    args = parser.parse_args()

    events = build_events(DATA_DIR / "cdaw_typeii_catalog.csv")
    events.to_csv(DATA_DIR / "typeii_events.csv", index=False)
    print(f"typeii_events.csv: {len(events)} onsets, {events.reaches_ip.sum()} reach IP, "
          f"{events.behind_limb.sum()} behind the limb")

    splits = build_splits(FULL_INDEX)
    (DATA_DIR / "splits").mkdir(exist_ok=True)
    for name, df in splits.items():
        df.to_csv(DATA_DIR / "splits" / f"{name}.csv", index=False)
    _summary(splits, events)

    if args.fetch_flares:
        flares = fetch_mx_flares()
        flares.to_csv(DATA_DIR / "mx_flares.csv", index=False)
        print(f"mx_flares.csv: {len(flares)} M/X flares")


if __name__ == "__main__":
    main()
