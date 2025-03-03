from typing import Tuple, Dict, Hashable, Optional, Any, List
import time
import datetime as dt
import numpy as np
from numpy import ndarray
import xarray as xr
from hydra import compose, initialize
from src.data.cpc_mrms_dataset import DailyAggregateRainfallDataset, get_precip_era5_dataset
from src.data.precip_dataloader_inference import RainfallSpecifiedInference
from src.utils.helper import deprecated, yprint, get_training_progressbar
# FIXME: for debug only. Remove later
from omegaconf import OmegaConf

__author__ = 'yuhao liu'


class RainfallDatasetCONUS(DailyAggregateRainfallDataset):
    """
    Precipitation dataset; loads the entire CONUS region without cropping.
    """

    def __init__(self, data_config: dict):
        super().__init__(data_config)

    def __getitem__(self, item):
        date_ = self.precip_dates[item]
        cpc_lr = self.read_cpc_daily_aggregate(date_)
        if self.use_precomputed_mrms:
            mrms = self.read_precomputed_mrms_daily_aggregate(date_)
        else:
            mrms = self.read_mrms_daily_aggregate(date_)
        cpc_lr = self.truncate_dataarray(cpc_lr)
        mrms = self.truncate_dataarray(mrms)
        if self.use_precomputed_era5:
            era5 = self.read_precomputed_era5_files(date_)
        else:
            era5 = self.read_era5_files(date_)

        if self.use_density:
            density = self.read_cpc_gauge_data(date_)
            ds, valid_mask = self.merge_all_dataarrays(cpc_lr, mrms, era5, density, date_=date_)
        else:
            ds, valid_mask = self.merge_all_dataarrays(cpc_lr, mrms, era5, date_=date_)
        batch, batch_coord = self.build_batch(ds) # MODIFIED, returns the entire xarray dataset, without cropping

        batch = self.correct_for_nan(batch)
        batch = self.normalize_precip(batch)
        batch = self.normalize_era5(batch)
        batch = self.normalize_density(batch)  # if needed
        batch = self.cvt_to_tensor(batch)
        # assume dimension matches
        batch_coord['date'] = date_
        return batch, batch_coord

    def build_batch(self, ds: xr.Dataset) -> tuple[dict[str | Hashable, ndarray], dict[str, ndarray]]:
        """
        Returns the full region in the same format as the cropped version.
        """
        # Return all data for each variable
        batch = {}
        for k in ds.data_vars:
            # Get the full array without slicing
            v = ds[k]
            # Rename keys as before
            if k == 'mrms':
                key = 'precip_gt'
            elif k == 'cpc':
                key = 'precip_up'
            else:
                key = k
            batch[key] = v.to_numpy()

        # Get the full coordinate extent using the first and last elements of the coordinates
        crop_coords = {
            'precip_lon_min': ds.coords['lon'].values[0],
            'precip_lon_max': ds.coords['lon'].values[-1],
            'precip_lat_min': ds.coords['lat'].values[0],
            'precip_lat_max': ds.coords['lat'].values[-1]
        }

        return batch, crop_coords

class RainfallDatasetNonSquare(RainfallSpecifiedInference):
    """
    Extension of the RainfallSpecifiedInference class to handle non-square data.
    """

    def __init__(self, data_config: dict, specify_eval_targets: Optional[List[Dict[str, Any]]] = None):
        shape_ = data_config.image_size
        assert len(shape_) == 2
        h, w = shape_
        max_len = max(h, w)
        data_config.image_size = max_len # override to be compatible with parent's init
        super().__init__(data_config, specify_eval_targets)
        self.img_size = shape_ # use actual image size for later use
        yprint(f"Using non-square image size: {self.img_size}. Requires tiled diffusion model.")
        return

    def select_crop(self, ds: xr.Dataset, location: dict) -> tuple[dict[str | Hashable, ndarray], dict[str, ndarray]]:
        """
        Overrides parent method to handle non-square crops.
        Returns a set of crops where ERA5 and GT precip have the same FoV. Raw ERA5 have lower resolution than GT precip.
        """
        lon, lat = location['lon'], location['lat']
        # Find the closest index in the DataArray to the given lon and lat
        lon_idx = np.abs(ds.coords['lon'] - lon).argmin().values.item()
        lat_idx = np.abs(ds.coords['lat'] - lat).argmin().values.item()
        # Calculate half sizes for slicing
        half_size_y, half_size_x = self.img_size[0] // 2, self.img_size[1] // 2
        # Calculate start and end indices for both dimensions
        start_lat_idx = lat_idx - half_size_y
        end_lat_idx = lat_idx + half_size_y
        start_lon_idx = lon_idx - half_size_x
        end_lon_idx = lon_idx + half_size_x

        # Select the sub-array using integer-location based indexing
        batch = {}
        for k in ds.data_vars:
            v = ds[k].isel(lat=slice(start_lat_idx, end_lat_idx), lon=slice(start_lon_idx, end_lon_idx))
            if k == 'mrms':
                k = 'precip_gt'
            elif k == 'cpc':
                k = 'precip_up'
            batch[k] = v.to_numpy()

        # get coordinates
        crop_coords = {
            'precip_lon_min': ds.coords['lon'].isel(lon=start_lon_idx).values,
            'precip_lon_max': ds.coords['lon'].isel(lon=end_lon_idx).values,
            'precip_lat_min': ds.coords['lat'].isel(lat=start_lat_idx).values,
            'precip_lat_max': ds.coords['lat'].isel(lat=end_lat_idx).values
        }

        return batch, crop_coords