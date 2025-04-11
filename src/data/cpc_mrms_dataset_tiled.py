from typing import Tuple, Dict, Hashable, Optional, Any, List
import time
import datetime as dt
import numpy as np
from numpy import ndarray
import xarray as xr
from hydra import compose, initialize
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from src.data.cpc_mrms_dataset import DailyAggregateRainfallDataset, get_precip_era5_dataset
from src.data.precip_dataloader_inference import RainfallSpecifiedInference
from src.utils.helper import deprecated, yprint, get_training_progressbar

__author__ = 'yuhao liu'


class RainfallDatasetNonSquare(RainfallSpecifiedInference):
    """
    Extension of the RainfallSpecifiedInference class to handle non-square data.
    """

    def __init__(self, data_config: dict, specify_eval_targets: Optional[List[Dict[str, Any]]] = None):
        shape_ = data_config.image_size
        if type(shape_) == ListConfig:
            h, w = shape_
            assert len(shape_) == 2, "Image size must be a list of 2 integers."
            max_len = max(h, w)
            data_config.image_size = max_len  # override to be compatible with parent's init
            super().__init__(data_config, specify_eval_targets)
            self.img_size = shape_  # use actual image size for later use
            yprint(f"Using non-square image size: {self.img_size}. Requires tiled diffusion model.")
        elif shape_ == 'full':
            data_config.image_size = 1  # override to be compatible with parent's init
            super().__init__(data_config, specify_eval_targets)
            self.img_size = 'full'
            yprint("Using full image size. Requires tiled diffusion model.")
        else:
            raise ValueError(f"Invalid image size: {shape_}")
        # data_config.image_size = 256  # FIXME: this should not be hardcoded restore the original image size for later use
        return

    def select_crop(self, ds: xr.Dataset, location: dict) -> tuple[dict[str | Hashable, ndarray], dict[str, ndarray]]:
        """
        Overrides parent method to handle special instructions
        * If the image size is square, use the parent method.
        * If the image size is non-square, use the select_nonsquare_crop method.
        * If the image size is 'full', return the full region.
        """
        if type(self.img_size) == int:
            yprint(f"Using square image size: {self.img_size}. There is a problem with the configuration."
                   f"Should have initialized with RainfallDatasetSquare instead.")
            return super().select_crop(ds, location)
        elif type(self.img_size) == ListConfig:
            assert len(self.img_size) == 2, "Image size must be a list of 2 integers."
            if self.img_size[0] == self.img_size[1]:
                return super().select_crop(ds, location)
            else:
                return self.select_nonsquare_crop(ds, location)
        elif self.img_size == 'full':
            return self.build_batch(ds)
        else:
            raise ValueError(f"Invalid image size: {self.img_size}")

    @staticmethod
    def build_batch(ds: xr.Dataset) -> tuple[dict[str | Hashable, ndarray], dict[str, ndarray]]:
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

    def select_nonsquare_crop(self, ds: xr.Dataset, location: dict) \
            -> tuple[dict[str | Hashable, ndarray], dict[str, ndarray]]:
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