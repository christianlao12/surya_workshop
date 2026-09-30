"""Tests for the models, loss, validation scores and Lightning module of the Type II app.

These pin the contracts the pieces rely on: both models return (B, 2) logits in LABELS
order, the loss is per-label BCE, and the validation scores are computed over the whole
epoch rather than per batch.
"""

import pytest
import torch
import torch.nn.functional as F

from tiny_models import (
    DEPTH,
    EMBED_DIM,
    IMG_SIZE,
    IN_CHANS,
    N_SPECTRAL_BLOCKS,
    PATCH_SIZE,
    make_batch,
)
from downstream_apps.radioburst.labels import LABELS
from downstream_apps.radioburst.lightning_modules.pl_simple_baseline import TypeIILightningModule
from downstream_apps.radioburst.metrics.radioburst_metrics import TypeIIMetrics, ValidationScores
from downstream_apps.radioburst.models.simple_baseline import LinearTypeIIModel
from workshop_infrastructure.configs import LoraAdapterConfig
from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
from workshop_infrastructure.utils import apply_peft_lora

MODES = ("train_loss", "val_loss", "train_metrics", "val_metrics")


def make_surya(num_outputs=len(LABELS)):
    return HelioSpectformer1D(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, embed_dim=EMBED_DIM,
        time_embedding={"type": "linear", "time_dim": 1}, depth=DEPTH,
        n_spectral_blocks=N_SPECTRAL_BLOCKS, num_heads=2, mlp_ratio=4, drop_rate=0.0,
        window_size=2, dp_rank=2, dtype=torch.float32, pooling="class_token",
        penultimate_linear_layer=True, num_outputs=num_outputs,
    )


def labelled_batch(batch_size=2):
    batch = make_batch(batch_size=batch_size)
    batch["labels"] = torch.tensor([[1.0, 1.0], [0.0, 0.0]])[:batch_size]
    return batch


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 2])
def test_surya_returns_two_logits_per_sample(batch_size):
    assert make_surya()(make_batch(batch_size=batch_size)).shape == (batch_size, 2)


def test_lora_keeps_the_two_output_head_trainable():
    model = apply_peft_lora(make_surya(), LoraAdapterConfig())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("head_unembed" in n for n in trainable)
    assert not any("backbone" in n and "lora_" not in n for n in trainable)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_baseline_returns_two_logits_per_sample(batch_size):
    model = LinearTypeIIModel(2 * IN_CHANS * 1)
    assert model(make_batch(batch_size=batch_size)).shape == (batch_size, 2)


def test_baseline_sees_spatial_spread_not_just_the_mean():
    torch.manual_seed(0)
    model = LinearTypeIIModel(2 * IN_CHANS)
    flat = torch.zeros(1, IN_CHANS, 1, 8, 8)
    checker = flat.clone()
    checker[..., ::2, ::2], checker[..., 1::2, 1::2] = 1.0, 1.0
    checker[..., ::2, 1::2], checker[..., 1::2, ::2] = -1.0, -1.0  # mean 0, std > 0
    assert not torch.allclose(model({"ts": flat}), model({"ts": checker}))


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------

def test_loss_is_per_label_bce_in_labels_order():
    logits = torch.tensor([[2.0, -1.0], [-0.5, 0.3]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    losses, weights = TypeIIMetrics("train_loss")(logits, labels)

    assert list(losses) == [f"bce_{n}" for n in LABELS]
    assert weights == [1.0, 1.0]
    for k, name in enumerate(LABELS):
        expected = F.binary_cross_entropy_with_logits(logits[:, k], labels[:, k])
        torch.testing.assert_close(losses[f"bce_{name}"], expected)


def test_weights_are_paired_with_their_own_term_and_recorded():
    metrics = TypeIIMetrics("train_loss", type2_weight=1.0, type2_ip_weight=0.25)
    _, weights = metrics(torch.zeros(2, 2), torch.zeros(2, 2))
    assert weights == [1.0, 0.25]
    assert metrics.loss_weights == {"type2_weight": 1.0, "type2_ip_weight": 0.25}


def test_val_loss_matches_the_training_objective():
    logits, labels = torch.randn(4, 2), torch.randint(0, 2, (4, 2)).float()
    train, _ = TypeIIMetrics("train_loss")(logits, labels)
    val, _ = TypeIIMetrics("val_loss")(logits, labels)
    for key in train:
        torch.testing.assert_close(train[key], val[key])


def test_per_batch_metric_modes_return_nothing():
    for mode in ("train_metrics", "val_metrics"):
        assert TypeIIMetrics(mode)(torch.zeros(2, 2), torch.zeros(2, 2)) == ({}, [])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        TypeIIMetrics("train")


# --------------------------------------------------------------------------------------
# Epoch-level validation scores
# --------------------------------------------------------------------------------------

def test_perfect_ranking_scores_one():
    scores = ValidationScores()
    labels = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    scores.update(labels * 4 - 2, labels)  # positives get logit +2, negatives -2
    out = scores.compute()
    for name in LABELS:
        for key in ("roc_auc", "pr_auc", "tss_max"):
            assert out[f"{name}_{key}"].item() == pytest.approx(1.0)
    assert out["type2_event_rate"].item() == pytest.approx(0.5)
    assert out["type2_ip_event_rate"].item() == pytest.approx(0.25)


def test_scores_accumulate_over_batches_like_one_big_batch():
    torch.manual_seed(0)
    logits = torch.randn(40, 2)
    labels = torch.stack([torch.arange(40) % 3 == 0, torch.arange(40) % 5 == 0], dim=1).float()

    whole = ValidationScores()
    whole.update(logits, labels)
    batched = ValidationScores()
    for i in range(0, 40, 2):  # batch size 2, as in training
        batched.update(logits[i:i + 2], labels[i:i + 2])

    expected, got = whole.compute(), batched.compute()
    assert expected.keys() == got.keys()
    for key in expected:
        torch.testing.assert_close(got[key], expected[key])


def test_undefined_scores_are_omitted_not_reported_as_zero():
    scores = ValidationScores()
    scores.update(torch.randn(4, 2), torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]))
    out = scores.compute()
    assert "type2_roc_auc" in out
    assert "type2_ip_roc_auc" not in out  # no IP positives in this epoch
    assert out["type2_ip_event_rate"].item() == 0.0


def test_reset_starts_a_fresh_epoch():
    scores = ValidationScores()
    scores.update(torch.randn(4, 2), torch.tensor([[1.0, 1.0], [0.0, 0.0]] * 2))
    scores.reset()
    scores.update(torch.randn(2, 2), torch.zeros(2, 2))
    assert scores.compute()["type2_event_rate"].item() == 0.0


# --------------------------------------------------------------------------------------
# Lightning module
# --------------------------------------------------------------------------------------

def make_module(model, **kwargs):
    metrics = {mode: TypeIIMetrics(mode) for mode in MODES}
    module = TypeIILightningModule(model, metrics, lr=1e-3, batch_size=2, **kwargs)
    logged = {}
    module.log = lambda name, value, **_: logged.__setitem__(name, value)  # no Trainer needed
    return module, logged


def test_training_step_returns_a_differentiable_loss_and_logs_components():
    module, logged = make_module(LinearTypeIIModel(2 * IN_CHANS))
    loss = module.training_step(labelled_batch(), 0)
    assert loss.ndim == 0 and loss.requires_grad
    loss.backward()
    assert {"train_loss", "train_loss_bce_type2", "train_loss_bce_type2_ip", "train_positives"} <= logged.keys()


def test_validation_epoch_logs_epoch_scores_and_resets():
    module, logged = make_module(LinearTypeIIModel(2 * IN_CHANS))
    module.validation_step(labelled_batch(), 0)
    module.validation_step(labelled_batch(), 1)
    module.on_validation_epoch_end()
    assert "val_loss" in logged
    assert "val_metric_type2_roc_auc" in logged and "val_metric_type2_event_rate" in logged
    assert module.val_scores.compute()["type2_event_rate"].isnan()  # reset: nothing accumulated


def test_preprocess_fn_is_applied_before_the_model():
    seen = {}

    def preprocess(batch):
        seen["called"] = True
        return batch

    module, _ = make_module(LinearTypeIIModel(2 * IN_CHANS), preprocess_fn=preprocess)
    module.training_step(labelled_batch(), 0)
    assert seen == {"called": True}


def test_run_info_and_loss_weights_are_recorded_as_hyperparameters():
    module, _ = make_module(LinearTypeIIModel(2 * IN_CHANS), run_info={"negative_keep_fraction": 0.2})
    assert module.hparams["negative_keep_fraction"] == 0.2
    assert module.hparams["type2_weight"] == 1.0


def test_one_lora_training_step_updates_the_head():
    torch.manual_seed(0)
    model = apply_peft_lora(make_surya(), LoraAdapterConfig())
    module, _ = make_module(model)
    head = {n: p.detach().clone() for n, p in model.named_parameters() if "head_unembed" in n and p.requires_grad}
    optimizer = module.configure_optimizers()
    module.training_step(labelled_batch(), 0).backward()
    optimizer.step()
    after = dict(model.named_parameters())
    assert any(not torch.equal(before, after[n]) for n, before in head.items())
