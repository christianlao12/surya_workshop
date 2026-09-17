"""Tests for the radio-burst fine-tuning head (HelioSpectformerBurst).

Checks the output contract the radio-burst metrics rely on, and that the extra bottleneck
decoder follows the head_ convention so it stays trainable under LoRA.
"""

import torch

from tiny_models import (
    DEPTH,
    EMBED_DIM,
    IMG_SIZE,
    IN_CHANS,
    N_SPECTRAL_BLOCKS,
    PATCH_SIZE,
    make_batch,
)
from downstream_apps.radioburst.metrics.radioburst_metrics import (
    RadioBurstMetrics,
    RadioBurstSpectraMetrics,
)
from downstream_apps.radioburst.models.finetune_burst import HelioSpectformerBurst
from workshop_infrastructure.configs import LoraAdapterConfig
from workshop_infrastructure.utils import apply_peft_lora, discover_head_modules

SPECTRUM_SHAPE = (3, 4)
RANK = 2


def make_burst_model():
    return HelioSpectformerBurst(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=EMBED_DIM,
        time_embedding={"type": "linear", "time_dim": 1},
        depth=DEPTH,
        n_spectral_blocks=N_SPECTRAL_BLOCKS,
        num_heads=2,
        mlp_ratio=4,
        drop_rate=0.0,
        window_size=2,
        dp_rank=2,
        dtype=torch.float32,
        spectrum_shape=SPECTRUM_SHAPE,
        spectra_rank=RANK,
    )


def make_target(batch_size=2):
    return {
        "burst": torch.tensor([1, 0][:batch_size]),
        "spectra": torch.randn(batch_size, *SPECTRUM_SHAPE),
    }


def make_preds(batch_size=2):
    """Stand-in head outputs, so the metric tests need no forward pass."""
    return {
        "burst_logit": torch.zeros(batch_size, 1),
        "spectra": torch.zeros(batch_size, *SPECTRUM_SHAPE),
    }


def test_head_layer_sizes():
    model = make_burst_model()
    assert model.head_unembed.out_features == 1
    coefficients, basis = model.head_spectra
    assert (coefficients.in_features, coefficients.out_features) == (EMBED_DIM, RANK)
    assert (basis.in_features, basis.out_features) == (
        RANK,
        SPECTRUM_SHAPE[0] * SPECTRUM_SHAPE[1],
    )


def test_output_keys_and_shapes():
    output = make_burst_model()(make_batch(batch_size=2))
    assert set(output) == {"burst_logit", "spectra"}
    assert output["burst_logit"].shape == (2, 1)
    assert output["spectra"].shape == (2, *SPECTRUM_SHAPE)


def test_output_shapes_for_batch_of_one():
    output = make_burst_model()(make_batch(batch_size=1))
    assert output["burst_logit"].shape == (1, 1)
    assert output["spectra"].shape == (1, *SPECTRUM_SHAPE)


def test_decoder_is_discovered_as_head():
    assert set(discover_head_modules(make_burst_model())) == {
        "head_cls_token",
        "head_linear",
        "head_unembed",
        "head_spectra",
    }


def test_lora_step_updates_both_output_layers():
    torch.manual_seed(0)
    model = apply_peft_lora(make_burst_model(), LoraAdapterConfig())

    tracked = {
        name: param
        for name, param in model.named_parameters()
        if "modules_to_save" in name
        and ("head_unembed" in name or "head_spectra" in name)
    }
    assert any("head_unembed" in n for n in tracked)
    assert any("head_spectra" in n for n in tracked)
    before = {name: param.detach().clone() for name, param in tracked.items()}

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    losses, weights = RadioBurstSpectraMetrics("train_loss")(model(make_batch()), make_target())
    sum(losses[k] * w for k, w in zip(losses, weights)).backward()
    optimizer.step()

    for name, param in tracked.items():
        assert not torch.equal(param, before[name]), f"{name} did not change"


def test_spectra_loss_is_masked_to_burst_rows():
    metrics = RadioBurstSpectraMetrics("train_loss")
    target = make_target()
    preds = {"burst_logit": torch.zeros(2, 1), "spectra": target["spectra"].clone()}
    preds["spectra"][1] += 100.0  # row 1 is quiet (burst == 0), so this must be ignored

    losses, _ = metrics(preds, target)
    assert losses["spectra_mse"] == 0
    assert torch.isfinite(losses["burst_bce"])


def test_missing_spectra_cells_are_skipped():
    """Real burst spectrograms contain scattered NaN cells; they must not poison the loss."""
    target = make_target()
    target["spectra"][0, 0, 0] = float("nan")
    preds = {"burst_logit": torch.zeros(2, 1), "spectra": torch.zeros(2, *SPECTRUM_SHAPE)}

    losses, _ = RadioBurstSpectraMetrics("train_loss")(preds, target)
    finite = target["spectra"][0][torch.isfinite(target["spectra"][0])]
    assert torch.isfinite(losses["spectra_mse"])
    assert torch.allclose(losses["spectra_mse"], (finite**2).mean())

    metrics, _ = RadioBurstSpectraMetrics("val_metrics")(preds, target)
    assert all(torch.isfinite(v) for v in metrics.values())


def test_default_weights_reproduce_the_unweighted_sum():
    """The defaults must not change the objective, so a reweighted run is attributable."""
    metrics = RadioBurstSpectraMetrics("train_loss")
    assert metrics.loss_weights == {"burst_weight": 1.0, "spectra_weight": 1.0}

    _, weights = metrics(make_preds(), make_target())
    assert list(weights) == [1.0, 1.0]


def test_weights_are_returned_paired_with_their_own_term():
    """The weight list is matched to the loss dict BY POSITION in _combine_losses.

    Nothing in that pairing is checked at runtime, so a term added to train_loss without a
    matching append would silently weight the wrong loss. This pins the mapping by name.
    """
    metrics = RadioBurstSpectraMetrics("train_loss", burst_weight=3.0, spectra_weight=0.25)
    losses, weights = metrics(make_preds(), make_target())

    assert dict(zip(losses, weights)) == {"burst_bce": 3.0, "spectra_mse": 0.25}
    assert metrics.loss_weights == {"burst_weight": 3.0, "spectra_weight": 0.25}


def test_val_loss_is_the_reweighted_sum():
    """val_loss must carry the same weights as train_loss: it is what ModelCheckpoint ranks."""
    preds, target = make_preds(), make_target()
    kwargs = dict(burst_weight=2.0, spectra_weight=0.5)

    train_losses, train_weights = RadioBurstSpectraMetrics("train_loss", **kwargs)(preds, target)
    val_losses, val_weights = RadioBurstSpectraMetrics("val_loss", **kwargs)(preds, target)

    assert list(val_losses) == list(train_losses)
    assert list(val_weights) == list(train_weights) == [2.0, 0.5]


def test_val_metrics_do_not_duplicate_val_loss_terms():
    """val_metrics used to recompute the masked MSE that val_loss already reports."""
    preds, target = make_preds(), make_target()

    losses, _ = RadioBurstSpectraMetrics("val_loss")(preds, target)
    metrics, weights = RadioBurstSpectraMetrics("val_metrics")(preds, target)

    assert set(losses).isdisjoint(metrics), "val_metrics re-reports a val_loss term"
    assert len(weights) == len(metrics)


def test_baseline_metric_names_match_the_spectra_head():
    """The linear baseline shares the <task>_<metric> naming, so notebook 1 and 2 line up."""
    preds = {"burst_prob": torch.full((2, 1), 0.5), "peak_amp": torch.zeros(2, 1)}
    target = {"burst": torch.tensor([1, 0]), "diagnostics": torch.zeros(2, 1)}

    losses, weights = RadioBurstMetrics("train_loss", burst_weight=2.0, peak_amp_weight=0.5)(
        preds, target
    )
    assert dict(zip(losses, weights)) == {"burst_bce": 2.0, "peak_amp_mse": 0.5}

    metrics, _ = RadioBurstMetrics("val_metrics")(preds, target)
    assert set(metrics) == {"burst_f1", "peak_amp_rrse"}
    assert set(losses).isdisjoint(metrics)
