import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Literal

from downstream_apps.radioburst.spectra_transform import read_spectra_file
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset


class RadioBurstDSDataset(HelioNetCDFDataset):
    """
    Template child class of HelioNetCDFDataset showing how to build a downstream dataset.
    Extends the base class with a burst label and radio spectra aligned to the Surya index.

    All ``HelioNetCDFDataset`` keyword arguments (``index_path``, ``scalers``, ``channels``,
    ``s3_cache_dir``, etc.) are accepted via ``**kwargs`` and forwarded to the base class.
    ``load_forecast_frames`` defaults to ``False`` here (the radio-burst catalog supplies
    its own labels, so future Surya frames are never fetched); pass it explicitly to override.

    Additional Args:
        return_surya_stack: If True (default), include the Surya image stack in the returned dict.
            Set to False to return only the burst label and spectra (useful for label inspection).
        max_number_of_samples: Cap the dataset length at this value. Useful for quick experiments.
        ds_radioburst_folder_path: Path to the folder containing the radio burst index and spectra.
        ds_radioburst_index_file: Filename of the radio burst CSV index, inside
            ``ds_radioburst_folder_path``.
        ds_time_column: Column name in the radio-burst catalog to use as the event timestamp.
        ds_forecast_horizon: Lead time between the Surya frame and the catalog timestamp
            (e.g., ``"3h"``). Catalog timestamps are shifted back by this amount before
            matching, so with ``ds_match_direction="forward"`` the Surya frame is at least
            this long before ``ds_time_column``. ``"0h"`` (default) matches without a lead.
        ds_time_tolerance: Maximum allowed time offset when matching Surya and DS indices
            (e.g., ``"15min"``), measured after the ``ds_forecast_horizon`` shift. With
            ``"forward"`` matching the lead time therefore lies in
            ``[ds_forecast_horizon, ds_forecast_horizon + ds_time_tolerance]``. Unmatched
            entries are dropped.
        ds_match_direction: Merge direction passed to ``pd.merge_asof``. Use ``"forward"``
            for causal prediction (predict bursts from prior solar state).
        ds_spectra_column: Location of the file of the spectra of the radio burst in the data folder.
        spectra_transform: Optional callable applied to the ``ds_spectra_column`` files (loaded and
            stacked into a ``pd.Series`` of arrays) to produce the ``normalized_spectra`` column.
            Signature: ``(spectra: pd.Series) -> pd.Series``. If ``None``, raw flux values are used
            as-is. Applied once, over the full catalog, before it is matched/split against the
            Surya index — so train and val ``RadioBurstDSDataset`` instances (each loading the same
            full catalog file) end up with identical statistics. Define this at the call site (see
            ``0_dataset_dataloader_template.ipynb``) to keep normalization logic out of the dataset
            class, mirroring ``label_transform`` in ``downstream_apps/template``.
        ds_diagnostics_columns: Optional list of catalog column names holding per-burst diagnostic
            measurements (e.g. ``["peak_amp", "energy", "f_centroid", "f_spread", "t_spread"]``) to
            expose as a regression target. If ``None`` (default), no diagnostics tuple is returned.
        ds_spectra_template_file: Optional filename, inside ``ds_radioburst_folder_path``, of a
            precomputed median burst-spectrogram template (see
            ``downstream_apps/radioburst/compute_median_template.py``). If given, loaded once
            into ``self.median_spectra_template`` — a ``(T, F)`` array with the leading
            non-value column dropped by position, matching the ``ds_spectra_column`` loader
            below. If ``None`` (default), ``self.median_spectra_template`` is ``None``.
    Raises:
        ValueError: If ``ds_radioburst_folder_path`` or ``ds_radioburst_index_file`` is not
            provided, or if no overlap exists between the Surya and DS indices within the
            specified tolerance.
    """

    def __init__(
        self,
        # Downstream-specific parameters
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        ds_radioburst_folder_path: str | None = None,
        ds_radioburst_index_file: str | None = None,
        ds_time_column: str | None = None,
        ds_forecast_horizon: str = "0h",
        ds_time_tolerance: str | None = None,
        ds_match_direction: Literal["forward", "backward", "nearest"] = "forward",
        ds_spectra_column: str | None = None,
        spectra_transform: Callable[[pd.Series], pd.Series] | None = None,
        ds_diagnostics_columns: list[str] | None = None,
        ds_spectra_template_file: str | None = None,
        # All HelioNetCDFDataset parameters (index_path, scalers, channels, s3_*, etc.)
        **kwargs,
    ):
        if ds_match_direction not in ["forward", "backward", "nearest"]:
            raise ValueError("ds_match_direction must be one of 'forward', 'backward', or 'nearest'")

        # load_forecast_frames defaults to False here: the radio-burst catalog supplies
        # its own labels, so future Surya frames never need to be fetched from disk/S3.
        kwargs.setdefault("load_forecast_frames", False)
        super().__init__(**kwargs)

        self.return_surya_stack = return_surya_stack

        # Load ds index and find intersection with Surya index
        if ds_radioburst_folder_path is None or ds_radioburst_index_file is None:
            raise ValueError("ds_radioburst_folder_path and ds_radioburst_index_file must be provided for RadioBurstDSDataset")
        self.ds_radioburst_folder_path = Path(ds_radioburst_folder_path)
        self.ds_spectra_column = ds_spectra_column
        self.ds_diagnostics_columns = ds_diagnostics_columns
        self.ds_index = pd.read_csv(self.ds_radioburst_folder_path / ds_radioburst_index_file)

        if self.ds_diagnostics_columns is not None:
            missing = set(self.ds_diagnostics_columns) - set(self.ds_index.columns)
            if missing:
                raise ValueError(
                    f"ds_diagnostics_columns not found in catalog: {sorted(missing)}"
                )

        # Shift each event back by the forecast horizon, so the Surya frame matched to it
        # below precedes the event by at least that much.
        self.ds_index["ds_index"] = pd.to_datetime(
            self.ds_index[ds_time_column]
        ).values.astype("datetime64[ns]") - pd.Timedelta(ds_forecast_horizon)
        self.ds_index.sort_values("ds_index", inplace=True)

        # Load every spectra file referenced by the full catalog and apply spectra_transform
        # once, here, over the whole column - before the merge_asof split below narrows
        # ds_index down to this particular phase's matched rows. Both the train and val
        # RadioBurstDSDataset instances load and transform this same full catalog file, so
        # they end up with identical statistics even though spectra_transform sees the
        # entire (train + val) population - mirroring how label_transform is applied to
        # FlareDSDataset's "intensity" column in downstream_apps/template.
        raw_spectra = self.ds_index[ds_spectra_column].apply(
            lambda p: read_spectra_file(self.ds_radioburst_folder_path, p)
        )
        if spectra_transform is not None:
            self.ds_index["normalized_spectra"] = spectra_transform(raw_spectra)
        else:
            self.ds_index["normalized_spectra"] = raw_spectra

        self.median_spectra_template = (
            pd.read_csv(self.ds_radioburst_folder_path / ds_spectra_template_file)
            .iloc[:, 1:]
            .to_numpy(dtype=np.float32)
            if ds_spectra_template_file
            else None
        )

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
                burst (np.int64): Burst label from the ``burst`` column (1 = burst
                    window, 0 = quiet window).
                spectra (np.ndarray[float32]): Radio spectra loaded from the file named in
                    ``ds_spectra_column``, shape (n_timesteps, n_bins), with the ``time``
                    column dropped, and passed through ``spectra_transform`` if one was given
                    at construction time (raw flux values otherwise).
                ds_index (str): ISO-format timestamp from the radioburst index.
                diagnostics (np.ndarray[float32]): Only present when ``ds_diagnostics_columns``
                    was given at construction time. Shape ``(len(ds_diagnostics_columns),)``.
                    Per-burst diagnostic measurements (e.g. peak amplitude, energy, frequency
                    centroid/spread, time spread) read from those catalog columns, in the given
                    order, as a regression target. Returned as a single array (not a tuple) so
                    the default ``DataLoader`` collate stacks samples into one ``(B, D)`` tensor
                    instead of transposing into ``D`` separate length-``B`` tensors.
                    ``NaN`` for quiet windows (``burst == 0``): the catalog has no burst to
                    diagnose there, so these entries are undefined, not missing data to impute.
                    Mask by ``burst`` before computing any loss over this target.
            When ``return_surya_stack=True``, also includes all keys from
            ``HelioNetCDFDataset.__getitem__`` (ts, time_delta_input, lead_time_delta, etc.).
        """
        sample = super().__getitem__(idx=idx) if self.return_surya_stack else {}
        row = self.df_valid_indices.iloc[idx]
        sample["burst"] = np.int64(row["burst"])
        sample["spectra"] = row["normalized_spectra"]
        if self.ds_diagnostics_columns is not None:
            sample["diagnostics"] = np.array(
                [row[col] for col in self.ds_diagnostics_columns], dtype=np.float32
            )
        sample["ds_index"] = self.df_valid_indices["ds_index"].iloc[idx].isoformat()
        return sample
