"""CPU-only tests for the radio-burst inference path.

These guard the three things that fail silently rather than loudly:
  * a checkpoint reloaded into a model whose head geometry does not match it,
  * a predicted spectrogram decoded with the wrong normalization constants,
  * a requested timestamp quietly answered with a different frame.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from tiny_models import EMBED_DIM, IMG_SIZE, IN_CHANS, PATCH_SIZE, make_batch

from downstream_apps.radioburst.inference import (
    find_catalog_window,
    head_geometry,
    load_baseline_model,
    predict_baseline,
    resolve_frame,
    single_frame_index,
    _strip_lightning_prefix,
)
from downstream_apps.radioburst.models.finetune_burst import HelioSpectformerBurst
from downstream_apps.radioburst.models.simple_baseline import TwoStageBurstModel
from downstream_apps.radioburst.spectra_transform import SpectraTemplateNormalizer
from workshop_infrastructure.configs import LoraAdapterConfig
from workshop_infrastructure.utils import apply_peft_lora

SPECTRUM_SHAPE = (3, 4)
SPECTRA_RANK = 2


def make_burst_model(spectrum_shape=SPECTRUM_SHAPE, spectra_rank=SPECTRA_RANK):
    """A HelioSpectformerBurst at the tiny sizes used across the test suite."""
    return HelioSpectformerBurst(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=EMBED_DIM,
        time_embedding={"type": "linear", "time_dim": 1},
        depth=3,
        n_spectral_blocks=1,
        num_heads=2,
        mlp_ratio=4,
        drop_rate=0.0,
        window_size=2,
        dp_rank=2,
        dtype=torch.float32,
        pooling="class_token",
        penultimate_linear_layer=True,
        spectrum_shape=spectrum_shape,
        spectra_rank=spectra_rank,
    )


def make_index_csv(path, timestamps, present=None):
    """Write a Surya-style index CSV with the three columns the dataset reads."""
    present = [1] * len(timestamps) if present is None else present
    pd.DataFrame(
        {
            "path": [f"s3://bucket/{i}.nc" for i in range(len(timestamps))],
            "timestep": timestamps,
            "present": present,
        }
    ).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------------------
# Checkpoint reload
# --------------------------------------------------------------------------------------


def test_head_geometry_is_read_from_the_checkpoint():
    model = apply_peft_lora(make_burst_model(), LoraAdapterConfig())
    spectra_rank, n_values = head_geometry(model.state_dict())
    assert spectra_rank == SPECTRA_RANK
    assert n_values == SPECTRUM_SHAPE[0] * SPECTRUM_SHAPE[1]


def test_head_geometry_rejects_a_checkpoint_without_the_burst_head():
    from tiny_models import make_model

    with pytest.raises(KeyError, match="HelioSpectformerBurst"):
        head_geometry(apply_peft_lora(make_model(), LoraAdapterConfig()).state_dict())


def test_strip_lightning_prefix_removes_only_the_wrapper_prefix():
    stripped = _strip_lightning_prefix({"model.base_model.weight": 1, "model.other": 2})
    assert stripped == {"base_model.weight": 1, "other": 2}
    # A state dict that is not Lightning-wrapped is returned untouched.
    assert _strip_lightning_prefix({"base_model.weight": 1}) == {"base_model.weight": 1}


def test_lora_checkpoint_round_trips_into_a_fresh_model():
    """A Lightning-style state dict reloads strictly and reproduces the same outputs."""
    torch.manual_seed(0)
    trained = apply_peft_lora(make_burst_model(), LoraAdapterConfig())
    checkpoint = {"state_dict": {f"model.{k}": v for k, v in trained.state_dict().items()}}

    torch.manual_seed(1)  # a differently initialized model, as at inference time
    reloaded = apply_peft_lora(make_burst_model(), LoraAdapterConfig())
    state_dict = _strip_lightning_prefix(checkpoint["state_dict"])
    reloaded.load_state_dict(state_dict, strict=True)  # strict: no silent partial load

    batch = make_batch(batch_size=2)
    trained.eval()
    reloaded.eval()
    with torch.no_grad():
        before, after = trained(batch), reloaded(batch)
    torch.testing.assert_close(before["burst_logit"], after["burst_logit"])
    torch.testing.assert_close(before["spectra"], after["spectra"])
    assert after["spectra"].shape == (2, *SPECTRUM_SHAPE)


# --------------------------------------------------------------------------------------
# Spectra normalization
# --------------------------------------------------------------------------------------


def make_raw_spectra(n=4, shape=(5, 3), seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series([(rng.random(shape) * 1e-13 + 1e-15).astype(np.float32) for _ in range(n)])


def make_template(shape=(5, 3), seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * 1e-13 + 1e-15).astype(np.float32)


def test_normalizer_inverse_recovers_the_raw_flux():
    raw = make_raw_spectra()
    normalizer = SpectraTemplateNormalizer.fit(raw, make_template())
    recovered = normalizer.inverse(np.stack(normalizer(raw).to_list()))
    np.testing.assert_allclose(recovered, np.stack(raw.to_list()), rtol=1e-5)


def test_normalizer_keeps_missing_cells_missing():
    raw = make_raw_spectra()
    raw.iloc[0][0, 0] = np.nan
    normalizer = SpectraTemplateNormalizer.fit(raw, make_template())
    normalized = normalizer(raw)
    assert np.isnan(normalized.iloc[0][0, 0])
    assert np.isnan(normalizer.inverse(normalized.iloc[0])[0, 0])
    assert np.isfinite(normalizer.center) and normalizer.scale > 0


def test_normalizer_survives_a_json_round_trip(tmp_path):
    normalizer = SpectraTemplateNormalizer.fit(make_raw_spectra(), make_template())
    reloaded = SpectraTemplateNormalizer.from_json(
        normalizer.to_json(tmp_path / "spectra_norm.json")
    )
    assert (reloaded.center, reloaded.scale) == (normalizer.center, normalizer.scale)
    np.testing.assert_array_equal(reloaded.log_template, normalizer.log_template)


def test_normalizer_rejects_a_non_positive_template():
    template = make_template()
    template[0, 0] = 0.0
    with pytest.raises(ValueError, match="finite and strictly positive"):
        SpectraTemplateNormalizer.fit(make_raw_spectra(), template)


def test_normalizer_rejects_a_spectrum_shape_that_does_not_match_the_template():
    normalizer = SpectraTemplateNormalizer.fit(make_raw_spectra(), make_template())
    mismatched = pd.Series([np.ones((5, 4), dtype=np.float32)])
    with pytest.raises(ValueError, match="does not match the template"):
        normalizer(mismatched)


# --------------------------------------------------------------------------------------
# peak_amp_scale: fit once from the catalog, not estimated per batch
# --------------------------------------------------------------------------------------


def make_random_spectra(n, shape, seed):
    rng = np.random.default_rng(seed)
    return [(rng.random(shape) * 1e-13 + 1e-15).astype(np.float32) for _ in range(n)]


def write_peak_amp_catalog(tmp_path, burst_spectra, quiet_spectra, shape):
    """Write a catalog (burst rows first, then quiet) + one spectra CSV per event + a
    template CSV, and return a cfg shaped like ``compute_peak_amp_scale`` expects.
    """
    rows = []
    for i, values in enumerate([*burst_spectra, *quiet_spectra]):
        df = pd.DataFrame(values, columns=[f"f{j}" for j in range(shape[1])])
        df.insert(0, "time", range(shape[0]))
        df.to_csv(tmp_path / f"spectra_{i}.csv", index=False)
        rows.append({"spectra_file": f"spectra_{i}.csv", "burst": int(i < len(burst_spectra))})
    pd.DataFrame(rows).to_csv(tmp_path / "catalog.csv", index=False)

    template = make_template(shape=shape, seed=99)
    template_df = pd.DataFrame(template, columns=[f"f{j}" for j in range(shape[1])])
    template_df.insert(0, "step", range(shape[0]))
    template_df.to_csv(tmp_path / "template.csv", index=False)

    return SimpleNamespace(
        data=SimpleNamespace(
            ds_radioburst_folder_path=str(tmp_path),
            ds_radioburst_index_file="catalog.csv",
            ds_spectra_column="spectra_file",
            ds_spectra_template_file="template.csv",
        )
    )


def test_compute_peak_amp_scale_matches_a_manual_computation(tmp_path):
    from downstream_apps.radioburst.spectra_transform import (
        compute_peak_amp_scale,
        load_catalog_spectra,
    )

    shape = (5, 3)
    cfg = write_peak_amp_catalog(
        tmp_path, make_random_spectra(4, shape, seed=1), make_random_spectra(4, shape, seed=2), shape
    )
    normalizer = SpectraTemplateNormalizer.fit_from_catalog(cfg)

    raw = load_catalog_spectra(
        cfg.data.ds_radioburst_folder_path, cfg.data.ds_radioburst_index_file, cfg.data.ds_spectra_column
    )
    catalog = pd.read_csv(tmp_path / "catalog.csv")
    normalized = normalizer(raw)
    expected_peaks = normalized[catalog["burst"] == 1].apply(np.nanmax)
    expected = max(float(np.nanvar(expected_peaks.to_numpy())), 1.0)

    assert compute_peak_amp_scale(cfg, normalizer) == pytest.approx(expected)


def test_compute_peak_amp_scale_ignores_quiet_rows(tmp_path):
    """Only burst==1 rows should move the scale — quiet rows' content must not matter."""
    from downstream_apps.radioburst.spectra_transform import compute_peak_amp_scale

    shape = (5, 3)
    burst = make_random_spectra(4, shape, seed=1)
    cfg = write_peak_amp_catalog(tmp_path, burst, make_random_spectra(4, shape, seed=2), shape)
    normalizer = SpectraTemplateNormalizer.fit_from_catalog(cfg)  # fit once, reused below
    before = compute_peak_amp_scale(cfg, normalizer)

    # Overwrite only the quiet rows' spectra files in place (indices 4..7); the already-
    # fitted normalizer and the burst rows' files are untouched.
    for i, values in enumerate(make_random_spectra(4, shape, seed=999)):
        df = pd.DataFrame(values, columns=[f"f{j}" for j in range(shape[1])])
        df.insert(0, "time", range(shape[0]))
        df.to_csv(tmp_path / f"spectra_{4 + i}.csv", index=False)

    assert compute_peak_amp_scale(cfg, normalizer) == pytest.approx(before)


def test_compute_peak_amp_scale_floors_at_one(tmp_path):
    """Identical burst rows give exactly-0 variance; the floor must keep it at 1.0."""
    from downstream_apps.radioburst.spectra_transform import compute_peak_amp_scale

    shape = (5, 3)
    identical = np.full(shape, 1e-13, dtype=np.float32)
    cfg = write_peak_amp_catalog(
        tmp_path, [identical.copy() for _ in range(4)], make_random_spectra(4, shape, seed=2), shape
    )
    normalizer = SpectraTemplateNormalizer.fit_from_catalog(cfg)

    assert compute_peak_amp_scale(cfg, normalizer) == pytest.approx(1.0)


def test_compute_peak_amp_scale_rejects_too_few_burst_rows(tmp_path):
    from downstream_apps.radioburst.spectra_transform import compute_peak_amp_scale

    shape = (5, 3)
    cfg = write_peak_amp_catalog(
        tmp_path, make_random_spectra(1, shape, seed=1), make_random_spectra(3, shape, seed=2), shape
    )
    normalizer = SpectraTemplateNormalizer.fit_from_catalog(cfg)
    with pytest.raises(ValueError, match="not enough"):
        compute_peak_amp_scale(cfg, normalizer)


# --------------------------------------------------------------------------------------
# Timestamp -> frame
# --------------------------------------------------------------------------------------


def test_resolve_frame_snaps_to_the_twelve_minute_grid(tmp_path):
    index = make_index_csv(tmp_path / "index.csv", ["2014-01-21 02:00:00", "2014-01-21 02:12:00"])
    assert resolve_frame("2014-01-21 02:03", index)["timestep"] == pd.Timestamp("2014-01-21 02:00")
    assert resolve_frame("2014-01-21 02:10", index)["timestep"] == pd.Timestamp("2014-01-21 02:12")


def test_resolve_frame_rejects_a_frame_that_is_not_present(tmp_path):
    index = make_index_csv(
        tmp_path / "index.csv",
        ["2014-01-21 02:00:00", "2014-01-21 02:12:00", "2014-01-21 02:24:00"],
        present=[1, 0, 1],
    )
    with pytest.raises(ValueError, match="before=2014-01-21 02:00:00, after=2014-01-21 02:24:00"):
        resolve_frame("2014-01-21 02:12", index)


def test_single_frame_index_writes_one_row_and_cleans_up(tmp_path):
    frame = resolve_frame(
        "2014-01-21 02:00", make_index_csv(tmp_path / "index.csv", ["2014-01-21 02:00:00"])
    )
    with single_frame_index(frame) as path:
        written = pd.read_csv(path)
        assert list(written.columns) == ["path", "timestep", "present"]
        assert len(written) == 1 and written["present"].iloc[0] == 1
    assert not path.exists()


def test_single_frame_index_cleans_up_after_an_error(tmp_path):
    frame = resolve_frame(
        "2014-01-21 02:00", make_index_csv(tmp_path / "index.csv", ["2014-01-21 02:00:00"])
    )
    with pytest.raises(RuntimeError):
        with single_frame_index(frame) as path:
            raise RuntimeError("boom")
    assert not path.exists()


# --------------------------------------------------------------------------------------
# The linear baseline
# --------------------------------------------------------------------------------------

BASELINE_INPUT_DIM = 2 * IN_CHANS  # mean and std per channel, one timestep


def make_baseline_checkpoint(tmp_path):
    """Save a TwoStageBurstModel the way Lightning does, and return (path, model)."""
    torch.manual_seed(0)
    model = TwoStageBurstModel(BASELINE_INPUT_DIM)
    path = tmp_path / "baseline-epoch=00-val_loss_bce=0.0000.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path)
    return path, model


def test_load_baseline_model_restores_weights(tmp_path):
    path, trained = make_baseline_checkpoint(tmp_path)

    reloaded = load_baseline_model(cfg=None, ckpt_path=path, device="cpu")

    batch = {"ts": torch.randn(2, IN_CHANS, 1, 8, 8)}
    trained.eval()
    with torch.no_grad():
        before, after = trained(batch), reloaded(batch)
    torch.testing.assert_close(before["peak_amp"], after["peak_amp"])
    torch.testing.assert_close(before["spectra"], after["spectra"])
    assert after["spectra"].shape == (2, 1, 1)


def test_load_baseline_model_rejects_a_finetuning_checkpoint(tmp_path):
    path = tmp_path / "wrong.ckpt"
    torch.save({"state_dict": {"model.head_unembed.weight": torch.zeros(1, 4)}}, path)
    with pytest.raises(KeyError, match="TwoStageBurstModel"):
        load_baseline_model(cfg=None, ckpt_path=path)


def make_scaler_cfg(factor=10.0):
    """A config and scalers whose inverse_transform is an obvious, checkable rescale."""
    channels = [f"c{i}" for i in range(IN_CHANS)]
    cfg = SimpleNamespace(data=SimpleNamespace(channels=channels))
    scalers = {c: SimpleNamespace(inverse_transform=lambda x, f=factor: x * f) for c in channels}
    return cfg, scalers


def test_predict_baseline_returns_a_probability_and_a_spectrogram(tmp_path):
    path, _ = make_baseline_checkpoint(tmp_path)
    model = load_baseline_model(cfg=None, ckpt_path=path)
    cfg, scalers = make_scaler_cfg()

    result = predict_baseline(model, make_batch(batch_size=2), cfg, scalers)

    assert result["burst_probability"].shape == (2,)
    assert ((0.0 <= result["burst_probability"]) & (result["burst_probability"] <= 1.0)).all()
    assert result["peak_amp"].shape == (2,)
    # "spectra" is peak_amp broadcast, still in standardized space: callers reconstruct
    # the full (T, F) raw-flux spectrogram via SpectraTemplateNormalizer.inverse().
    assert result["spectra"].shape == (2, 1, 1)


def test_predict_baseline_destandardizes_its_input(tmp_path):
    """The baseline reads signum-log space; skipping the inverse z-score changes the answer."""
    path, _ = make_baseline_checkpoint(tmp_path)
    model = load_baseline_model(cfg=None, ckpt_path=path)
    batch = make_batch(batch_size=2)

    cfg, scalers = make_scaler_cfg(factor=10.0)
    scaled = predict_baseline(model, batch, cfg, scalers)["peak_amp"]
    cfg_identity, identity = make_scaler_cfg(factor=1.0)
    unscaled = predict_baseline(model, batch, cfg_identity, identity)["peak_amp"]

    assert not np.allclose(scaled, unscaled)


# --------------------------------------------------------------------------------------
# Matching a forecast window to the catalog
# --------------------------------------------------------------------------------------


def make_catalog_cfg(tmp_path, windows, tolerance="30m"):
    """Write a tiny catalog of (window_start, burst) rows and return a config for it."""
    pd.DataFrame(
        {
            "window_start": [w for w, _ in windows],
            "burst": [b for _, b in windows],
            "window_start_file": [f"burst_data/{i}.csv" for i in range(len(windows))],
        }
    ).to_csv(tmp_path / "catalog.csv", index=False)
    return SimpleNamespace(
        data=SimpleNamespace(
            ds_radioburst_folder_path=str(tmp_path),
            ds_radioburst_index_file="catalog.csv",
            ds_time_column="window_start",
            ds_time_tolerance=tolerance,
        )
    )


def test_find_catalog_window_matches_the_window_it_was_trained_against(tmp_path):
    cfg = make_catalog_cfg(tmp_path, [("2014-01-21 05:00:00", 1), ("2019-01-29 16:00:00", 0)])
    assert find_catalog_window(cfg, "2014-01-21 05:00")["burst"] == 1
    assert find_catalog_window(cfg, "2019-01-29 16:00")["burst"] == 0


def test_find_catalog_window_takes_the_nearest_row_forward_not_the_first_listed(tmp_path):
    cfg = make_catalog_cfg(tmp_path, [("2014-01-21 05:20:00", 1), ("2014-01-21 05:05:00", 0)])
    assert find_catalog_window(cfg, "2014-01-21 05:00")["burst"] == 0  # the 05:05 row


def test_find_catalog_window_includes_the_tolerance_boundary(tmp_path):
    cfg = make_catalog_cfg(tmp_path, [("2014-01-21 05:30:00", 1)], tolerance="30m")
    assert find_catalog_window(cfg, "2014-01-21 05:00") is not None
    assert find_catalog_window(cfg, "2014-01-21 04:59") is None      # 31 min away


def test_find_catalog_window_ignores_windows_already_past(tmp_path):
    cfg = make_catalog_cfg(tmp_path, [("2014-01-21 04:00:00", 1)])
    assert find_catalog_window(cfg, "2014-01-21 05:00") is None


def test_find_catalog_window_defaults_to_the_configs_tolerance(tmp_path):
    cfg = make_catalog_cfg(tmp_path, [("2014-01-21 06:00:00", 1)], tolerance="30m")
    assert find_catalog_window(cfg, "2014-01-21 05:00") is None
    assert find_catalog_window(cfg, "2014-01-21 05:00", tolerance="2h") is not None
