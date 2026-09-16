"""Tests for the radio-burst fine-tuning head (HelioSpectformerBurst).

Checks the output contract the radio-burst metrics rely on, and that the extra bottleneck
decoder follows the head_ convention so it stays trainable under LoRA.
"""

import torch
from torch import nn

from conftest import (
    DEPTH,
    EMBED_DIM,
    IMG_SIZE,
    IN_CHANS,
    N_SPECTRAL_BLOCKS,
    PATCH_SIZE,
    make_batch,
)
from downstream_apps.radioburst.metrics.radioburst_metrics import RadioBurstSpectraMetrics
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


def test_head_layer_sizes():
    model = make_burst_model()
    assert model.head_unembed.out_features == 1 + RANK
    assert isinstance(model.head_spectra_decoder, nn.Linear)
    assert model.head_spectra_decoder.in_features == RANK
    assert model.head_spectra_decoder.out_features == SPECTRUM_SHAPE[0] * SPECTRUM_SHAPE[1]


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
        "head_spectra_decoder",
    }


def test_lora_step_updates_both_output_layers():
    torch.manual_seed(0)
    model = apply_peft_lora(make_burst_model(), LoraAdapterConfig())

    tracked = {
        name: param
        for name, param in model.named_parameters()
        if "modules_to_save" in name
        and ("head_unembed" in name or "head_spectra_decoder" in name)
    }
    assert any("head_unembed" in n for n in tracked)
    assert any("head_spectra_decoder" in n for n in tracked)
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
    assert losses["mse_spectra"] == 0
    assert torch.isfinite(losses["bce"])


def test_missing_spectra_cells_are_skipped():
    """Real burst spectrograms contain scattered NaN cells; they must not poison the loss."""
    target = make_target()
    target["spectra"][0, 0, 0] = float("nan")
    preds = {"burst_logit": torch.zeros(2, 1), "spectra": torch.zeros(2, *SPECTRUM_SHAPE)}

    losses, _ = RadioBurstSpectraMetrics("train_loss")(preds, target)
    finite = target["spectra"][0][torch.isfinite(target["spectra"][0])]
    assert torch.isfinite(losses["mse_spectra"])
    assert torch.allclose(losses["mse_spectra"], (finite**2).mean())

    metrics, _ = RadioBurstSpectraMetrics("val_metrics")(preds, target)
    assert all(torch.isfinite(v) for v in metrics.values())
