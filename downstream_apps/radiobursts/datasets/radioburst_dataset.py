import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Literal
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset


class RadioBurstDSDataset(HelioNetCDFDataset):
    """
    Template child class of HelioNetCDFDataset showing how to build a downstream dataset.
    Extends the base class with a flare intensity label aligned to the Surya index.

    All ``HelioNetCDFDataset`` keyword arguments (``index_path``, ``scalers``, ``channels``,
    ``s3_cache_dir``, etc.) are accepted via ``**kwargs`` and forwarded to the base class.
    ``load_forecast_frames`` defaults to ``False`` here (flare forecasting supplies its own
    labels, so future Surya frames are never fetched); pass it explicitly to override.

    Additional Args:
        return_surya_stack: If True (default), include the Surya image stack in the returned dict.
            Set to False to return only the flare intensity label (useful for label inspection).
        max_number_of_samples: Cap the dataset length at this value. Useful for quick experiments.
        radioburst_folder_path: Path to the folder containing the radio burst index and spectra.
        radioburst_index_path: Path to the radio burst CSV index.
        ds_time_column: Column name in the flare index to use as the event timestamp.
        ds_time_tolerance: Maximum allowed time offset when matching Surya and DS indices
            (e.g., ``"15min"``). Unmatched entries are dropped.
        ds_match_direction: Merge direction passed to ``pd.merge_asof``. Use ``"forward"``
            for causal prediction (predict flares from prior solar state).
        ds_spectra_column: Location of the file of the spectra of the radio burst in the data folder.
    Raises:
        ValueError: If ``ds_flare_index_path`` is not provided, or if no overlap exists
            between the Surya and DS indices within the specified tolerance.
    """

    def __init__(
        self,
        # Downstream-specific parameters
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        ds_radioburst_folder_path: str | None = None,
        ds_radioburst_index_file: str | None = None,
        ds_time_column: str | None = None,
        ds_time_tolerance: str | None = None,
        ds_match_direction: Literal["forward", "backward", "nearest"] = "forward",
        ds_spectra_column: str | None = None,
        # All HelioNetCDFDataset parameters (index_path, scalers, channels, s3_*, etc.)
        **kwargs,
    ):
        if ds_match_direction not in ["forward", "backward", "nearest"]:
            raise ValueError("ds_match_direction must be one of 'forward', 'backward', or 'nearest'")

        # load_forecast_frames defaults to False here: flare forecasting supplies its
        # own labels, so future Surya frames never need to be fetched from disk/S3.
        kwargs.setdefault("load_forecast_frames", False)
        super().__init__(**kwargs)

        self.return_surya_stack = return_surya_stack

        # Load ds index and find intersection with Surya index
        if ds_radioburst_folder_path is None or ds_radioburst_index_file is None:
            raise ValueError("ds_radioburst_folder_path and ds_radioburst_index_file must be provided for RadioBurstDSDataset")
        self.ds_radioburst_folder_path = Path(ds_radioburst_folder_path)
        self.ds_spectra_column = ds_spectra_column
        self.ds_index = pd.read_csv(self.ds_radioburst_folder_path / ds_radioburst_index_file)

        self.ds_index["ds_index"] = pd.to_datetime(
            self.ds_index[ds_time_column]
        ).values.astype("datetime64[ns]")
        self.ds_index.sort_values("ds_index", inplace=True)

        # Create Surya valid indices and find closest match to DS index
        self.df_valid_indices = pd.DataFrame(
            {"valid_indices": self.valid_indices}
        ).sort_values("valid_indices")
        self.df_valid_indices = pd.merge_asof(
            self.df_valid_indices,
            self.ds_index,
            right_on="ds_index",
            left_on="valid_indices",
            direction=ds_match_direction,
        )
        # Remove duplicates keeping closest match
        self.df_valid_indices["index_delta"] = np.abs(
            self.df_valid_indices["valid_indices"] - self.df_valid_indices["ds_index"]
        )
        self.df_valid_indices = self.df_valid_indices.sort_values(
            ["ds_index", "index_delta"]
        )
        self.df_valid_indices.drop_duplicates(
            subset="ds_index", keep="first", inplace=True
        )
        # Enforce a maximum time tolerance for matches
        if ds_time_tolerance is not None:
            self.df_valid_indices = self.df_valid_indices.loc[
                self.df_valid_indices["index_delta"] <= pd.Timedelta(ds_time_tolerance),
                :,
            ]
            if len(self.df_valid_indices) == 0:
                raise ValueError("No intersection between Surya and DS indices")

        # Override valid indices variables to reflect matches between Surya and DS
        self.valid_indices = [
            pd.Timestamp(date) for date in self.df_valid_indices["valid_indices"]
        ]
        self.adjusted_length = len(self.valid_indices)
        self.df_valid_indices.set_index("valid_indices", inplace=True)

        if max_number_of_samples is not None and max_number_of_samples < self.adjusted_length:
            self.valid_indices = self.valid_indices[:max_number_of_samples]
            self.df_valid_indices = self.df_valid_indices.iloc[:max_number_of_samples]
            self.adjusted_length = max_number_of_samples

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Dataset index.

        Returns:
            Dictionary containing:
                forecast_0 (np.int64): Burst label from the ``burst`` column (1 = burst
                    window, 0 = quiet window).
                forecast_1 (np.ndarray[float32]): Radio spectra loaded from the file named in
                    ``ds_spectra_column``, shape (n_timesteps, n_bins), with the ``time``
                    column dropped.
                ds_index (str): ISO-format timestamp from the radioburst index.
            When ``return_surya_stack=True``, also includes all keys from
            ``HelioNetCDFDataset.__getitem__`` (ts, time_delta_input, lead_time_delta, etc.).
        """
        sample = super().__getitem__(idx=idx) if self.return_surya_stack else {}
        spectra_path = self.ds_radioburst_folder_path / self.df_valid_indices.iloc[idx][self.ds_spectra_column]
        spectra_df = pd.read_csv(spectra_path)
        sample["forecast_0"] = np.int64(self.df_valid_indices.iloc[idx]["burst"])
        sample["forecast_1"] = spectra_df.drop(columns="time").to_numpy(dtype=np.float32)
        sample["ds_index"] = self.df_valid_indices["ds_index"].iloc[idx].isoformat()
        return sample
