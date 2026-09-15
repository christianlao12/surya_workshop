"""
Task-specific configuration for the flare-forecasting template app.

Everything generic — paths, channels, temporal sampling, S3 settings, the model and
LoRA configs, the training and logging sections, and ``load_config()`` itself — lives in
``workshop_infrastructure/configs.py``. This file holds only what is specific to *this*
task: the flare catalog and how its events are aligned to the Surya index.

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
class RadioBurstDataConfig(DataConfig):
    """DataConfig plus the flare-catalog alignment settings used by ``RadioBurstDSDataset``.

    These four keys are what makes this app's ``data:`` section different from any other
    downstream task's. Swap them for your own when you fork.
    """
    # Path to the data folder
    ds_radioburst_folder_path: str = ""
    # Filename of the label catalog *inside* ds_radioburst_folder_path. Not a PATH_FIELD:
    # RadioBurstDSDataset joins it onto ds_radioburst_folder_path itself, so resolving it
    # independently here (relative to the config file's dir) would produce the wrong path.
    ds_radioburst_index_file: str = ""
    # Column in the catalog holding the event timestamp.
    ds_time_column: str = "window_start"
    # Max allowed gap when matching catalog events to Surya timesteps.
    ds_time_tolerance: str = "1h"
    # "forward" uses the solar state *before* the flare (causal prediction).
    ds_match_direction: str = "forward"
    # Column in catalog pointing to spectra file path
    ds_spectra_column: str = "window_start_file"

    # Only the folder is a standalone path needing the same relative-to-the-config-file
    # resolution as the base class's fields. ds_radioburst_index_file is deliberately
    # excluded (see comment above).
    PATH_FIELDS: ClassVar[tuple[str, ...]] = DataConfig.PATH_FIELDS + ("ds_radioburst_folder_path",)


# The app's entry point. Identical to load_config() except that the data: section is
# parsed into FlareDataConfig, so the four keys above are recognized instead of rejected.
load_radioburst_config = partial(load_config, data_cls=RadioBurstDataConfig)


__all__ = [
    "RadioBurstDataConfig",
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
