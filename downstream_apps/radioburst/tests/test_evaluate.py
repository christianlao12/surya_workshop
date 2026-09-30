"""Tests for evaluate.py: the scores, the probability correction, and checkpoint reloading.

The scoring functions decide the headline result, so each is checked against a hand
calculation. They need no torch; the checkpoint test is skipped where PEFT is missing.
"""

import numpy as np
import pandas as pd
import pytest

from downstream_apps.radioburst import evaluate as ev
from downstream_apps.radioburst.prepare_data import ROTATION, START


# --------------------------------------------------------------------------------------
# Probability correction for subsampled negatives
# --------------------------------------------------------------------------------------

def test_keeping_every_negative_needs_no_correction():
    p = np.array([0.01, 0.3, 0.9])
    np.testing.assert_allclose(ev.correct_for_subsampling(p, 1.0), p)


def test_correction_multiplies_the_odds_by_the_kept_fraction():
    # p = 0.5 -> odds 1; keeping 10% of negatives -> true odds 0.1 -> p = 0.1 / 1.1
    assert ev.correct_for_subsampling(0.5, 0.1) == pytest.approx(0.1 / 1.1)


def test_correction_recovers_the_true_rate_of_a_subsampled_population():
    """A model that learned the subsampled base rate exactly is mapped back to the true one."""
    true_rate, keep = 0.05, 0.2
    subsampled_rate = true_rate / (true_rate + (1 - true_rate) * keep)
    assert ev.correct_for_subsampling(subsampled_rate, keep) == pytest.approx(true_rate)


# --------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------

def test_tss_by_hand():
    y = [1, 1, 0, 0, 0, 0]
    forecast = [1, 0, 1, 0, 0, 0]  # hit rate 1/2, false alarm rate 1/4
    assert ev.tss(y, forecast) == pytest.approx(0.25)


def test_tss_is_undefined_without_both_classes():
    assert np.isnan(ev.tss([0, 0, 0], [1, 0, 0]))
    assert np.isnan(ev.tss([1, 1], [1, 0]))


def test_best_threshold_separates_a_perfectly_ranked_forecast():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7])
    threshold = ev.best_threshold(y, p)
    assert ev.tss(y, p >= threshold) == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.6


def test_brier_skill_is_zero_for_the_reference_and_one_for_perfect():
    y = np.array([0, 0, 0, 1])
    assert ev.brier_skill(y, np.full(4, 0.25), 0.25) == pytest.approx(0.0)
    assert ev.brier_skill(y, y.astype(float), 0.25) == pytest.approx(1.0)


def test_rotations_are_counted_from_start():
    times = [START, START + ROTATION - pd.Timedelta("1h"), START + ROTATION]
    assert ev.rotation_of(times).tolist() == [0, 0, 1]


def test_bootstrap_resamples_whole_rotations():
    """Every resample holds whole rotations: never part of one."""
    times = START + ROTATION * np.repeat(np.arange(6), 4) + pd.Timedelta("1D")
    sizes = []
    ev.bootstrap_interval(lambda idx: sizes.append(len(idx)) or 0.0, pd.DatetimeIndex(times), n=50)
    assert all(size % 4 == 0 for size in sizes)


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    times = pd.date_range(START, periods=400, freq="6h")
    y = rng.random(400) < 0.2
    f = np.where(rng.random(400) < 0.7, y, ~y)
    statistic = lambda idx: ev.tss(y[idx], f[idx])  # noqa: E731
    first = ev.bootstrap_interval(statistic, times, n=200, seed=3)
    assert first == ev.bootstrap_interval(statistic, times, n=200, seed=3)
    assert first[0] <= ev.tss(y, f) <= first[1]


def test_forecast_compared_with_itself_has_zero_dtss():
    rng = np.random.default_rng(0)
    times = pd.date_range(START, periods=200, freq="6h")
    y = (rng.random(200) < 0.2).astype(int)
    forecast = rng.integers(0, 2, 200)
    row = ev.score_forecast(y, forecast, times, None, 0.2, probabilistic=False,
                            reference_binary=forecast, n_boot=50)
    assert row["dtss_vs_mx"] == 0 and row["dtss_ci_low"] == 0 and row["dtss_ci_high"] == 0
    assert np.isnan(row["bss"])  # 0/1 forecasts get no probabilistic scores


# --------------------------------------------------------------------------------------
# Reference forecasts
# --------------------------------------------------------------------------------------

def test_reference_forecasts_look_only_at_the_past(tmp_path):
    t0 = pd.Timestamp("2021-03-01 00:00")
    flares = tmp_path / "flares.csv"
    pd.DataFrame({"start_time": [t0 - pd.Timedelta("2h"), t0 + pd.Timedelta("3h")]}).to_csv(flares, index=False)
    cfg = type("Cfg", (), {"data": type("D", (), {"ds_flares_path": str(flares), "ds_horizon": "24h"})})
    labels = pd.DataFrame({"type2": [0.0, 0.0]}, index=pd.DatetimeIndex([t0, t0 - pd.Timedelta("6h")]))
    events = pd.DataFrame({"onset": [t0 - pd.Timedelta("1h")], "reaches_ip": [0]})

    refs = ev.reference_forecasts(cfg, labels, events, {"type2": 0.05, "type2_ip": 0.03})
    # At t0 the flare 2 h earlier counts; the one 3 h later must not. At t0-6h neither has happened.
    assert refs["mx_flare_24h"]["type2"].tolist() == [1.0, 0.0]
    assert refs["persistence"]["type2"].tolist() == [1.0, 0.0]
    assert refs["persistence"]["type2_ip"].tolist() == [0.0, 0.0]  # the onset is not IP
    assert refs["climatology"]["type2"].tolist() == [0.05, 0.05]


# --------------------------------------------------------------------------------------
# Checkpoint reloading (needs torch + peft: runs on the training machine)
# --------------------------------------------------------------------------------------

def test_strip_lightning_prefix_keeps_only_model_weights():
    state = {"model.base_model.w": 1, "model.head_unembed.b": 2, "val_scores.x": 3}
    assert ev.strip_lightning_prefix(state) == {"base_model.w": 1, "head_unembed.b": 2}


def test_lora_checkpoint_round_trips_into_a_fresh_model():
    pytest.importorskip("peft")
    import torch
    from tiny_models import make_batch, make_model
    from workshop_infrastructure.configs import LoraAdapterConfig
    from workshop_infrastructure.utils import apply_peft_lora

    torch.manual_seed(0)
    trained = apply_peft_lora(make_model(num_outputs=2), LoraAdapterConfig())
    state = {f"model.{k}": v for k, v in trained.state_dict().items()}

    torch.manual_seed(1)  # a differently initialized model, as at evaluation time
    reloaded = apply_peft_lora(make_model(num_outputs=2), LoraAdapterConfig())
    reloaded.load_state_dict(ev.strip_lightning_prefix(state), strict=True)

    batch = make_batch(batch_size=2)
    with torch.no_grad():
        torch.testing.assert_close(trained.eval()(batch), reloaded.eval()(batch))
