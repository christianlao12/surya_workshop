"""
Task-specific configuration for the Type II radio-burst forecasting app.

Everything generic — paths, channels, temporal sampling, S3 settings, the model and
LoRA configs, the training and logging sections, and ``load_config()`` itself — lives in
``workshop_infrastructure/configs.py``. This file holds only what is specific to *this*
task: where the Type II events live and how labels are drawn from them.

**This is the pattern to copy when you fork the template.** Subclass ``DataConfig`` with
your task's fields, then bind ``load_config`` to it. You never maintain a copy of the
base config.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import ClassVar

from workshop_infrastructure.configs import (  # re-exported for convenience
    DataConfig,
    LoraAdapterConfig,
    ModelConfig,
    OutputConfig,
    TimeEmbeddingConfig,
    TrainingConfig,
    load_config,
)


@dataclass
class TypeIIDataConfig(DataConfig):
    """DataConfig plus the Type II label settings used by ``TypeIIDataset``.

    ``train_data_path``/``valid_data_path`` (inherited) point at the split files written by
    ``prepare_data.py``; ``test_data_path`` is read only by ``evaluate.py``.
    """
    # Held-out cycle-25 split. Never used for training or checkpoint selection.
    test_data_path: str = ""
    # One row per Type II onset (data/typeii_events.csv, written by prepare_data.py).
    ds_events_path: str = ""
    # M/X flare list for the "flare in the last 24 h" baseline (evaluate.py only).
    ds_flares_path: str = ""
    # Label window: a sample at issue time t is positive if an onset falls in (t, t + horizon].
    # Must match HORIZON in prepare_data.py, which sizes the gap between splits.
    ds_horizon: str = "24h"
    # Keep this many negatives per positive in the *train* split (null = keep all).
    # Validation and test always keep the natural event rate.
    ds_negative_ratio: float | None = None
    # Drop samples whose label window contains a behind-the-limb onset (invisible to SDO).
    ds_exclude_behind_limb: bool = False

    PATH_FIELDS: ClassVar[tuple[str, ...]] = DataConfig.PATH_FIELDS + (
        "test_data_path",
        "ds_events_path",
        "ds_flares_path",
    )


# The app's entry point. Identical to load_config() except that the data: section is
# parsed into TypeIIDataConfig, so the fields above are recognized instead of rejected.
load_radioburst_config = partial(load_config, data_cls=TypeIIDataConfig)


__all__ = [
    "TypeIIDataConfig",
    "load_radioburst_config",
    # Re-exports so app code can import everything config-related from one place.
    "DataConfig",
    "OutputConfig",
    "TrainingConfig",
    "ModelConfig",
    "LoraAdapterConfig",
    "TimeEmbeddingConfig",
    "load_config",
]
