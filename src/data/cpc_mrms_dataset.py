import os
import os.path as p
from typing import Tuple, Dict, Hashable
from hydra import compose, initialize
import numpy as np
import xarray as xr
import torch
from numpy import ndarray
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from natsort import natsorted
import time
import datetime as dt
import re
import pandas as pd
from matplotlib import pyplot as plt
from src.utils.helper import deprecated, yprint, rprint, InfiniteLoader, get_training_progressbar
from src.utils.cpc_utils import read_cpc_file_from, read_cpc_gz_file_from, generate_mrms_dailyagg_12z

# FIXME: for debug only. Remove later
from omegaconf import OmegaConf

__author__ = 'yuhao liu'


def get_precip_era5_dataset(cfg, eval_num_worker=None, eval_batch_size=None):
    print('------ Dataset statistics -------')
    if cfg.data.data_config.uniform_dequantization:
        raise NotImplementedError('Uniform dequantization not yet supported.')
    # if config.mode != 'train':
    #     raise NotImplementedError('Evaluation not yet supported.')
    train_val_dataset = DailyAggregateRainfallDataset(cfg.data.data_config)
    dataset_size = len(train_val_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(cfg.data.train_val_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    generator = torch.Generator().manual_seed(cfg.seed)
    train_sampler = SubsetRandomSampler(train_indices, generator=generator)
    val_sampler = SubsetRandomSampler(val_indices, generator=generator)

    train_loader = DataLoader(train_val_dataset, batch_size=cfg.data.batch_size,  # persistent_workers=True,
                              timeout=0,  # 120,
                              num_workers=cfg.data.num_workers, sampler=train_sampler)
                              # num_workers=0, sampler=train_sampler)

    val_loader = DataLoader(train_val_dataset, batch_size=eval_batch_size if eval_batch_size is not None else cfg.data.batch_size,
                            timeout=0,  # 120,
                            num_workers=eval_num_worker if eval_num_worker is not None else cfg.data.num_workers,
                            sampler=val_sampler)
    print('Split dataset into {} training samples and {} validation samples'
          .format(len(train_indices), len(val_indices)))
    print(f'Training dataset contains data from '
          f'{train_val_dataset.precip_dates[split]} to {train_val_dataset.precip_dates[-1]}')
    print(f'Validation dataset contains data from '
          f'{train_val_dataset.precip_dates[0]} to {train_val_dataset.precip_dates[split - 1]}')
    print('---------------------------------')
    return train_loader, val_loader, train_val_dataset


class DailyAggregateRainfallDataset(Dataset):

    def __init__(self, data_config: dict):
        super()
        # check config
        # assert config.data.condition_mode in [2, 6], 'Condition mode must be 2 or 6'

        # self.high_res_dim = (config.data.image_size, config.data.image_size)
        self.img_size = data_config.image_size

        # paths
        self.cpc_aggregate_path = data_config.dataset_path.cpc
        self.use_precomputed_mrms = data_config.use_precomputed_daily_aggregates
        self.use_precomputed_era5 = data_config.use_precomputed_era5
        self.use_precomputed_cpc = data_config.use_precomputed_cpc
        if self.use_precomputed_mrms:
            yprint('Using precomputed MRMS daily aggregates')
            self.mrms_path = data_config.dataset_path.mrms_daily
        else:
            yprint('Computing MRMS on the fly')
            self.mrms_path = data_config.dataset_path.mrms
        if self.use_precomputed_era5:
            yprint('Using precomputed ERA5 daily aggregates')
            self.era5_path = data_config.dataset_path.interpolated_era5
            self.era5_filenames = data_config.interpolated_era5_filenames
        else:
            yprint('Computing ERA5 on the fly')
            self.era5_path = data_config.dataset_path.era5
            self.era5_filenames = data_config.era5_filenames
        if self.use_precomputed_cpc:
            yprint('Using precomputed CPC daily aggregates')
            self.cpc_gauge_density_path = data_config.dataset_path.cpc_gauge
            self.cpc_aggregate_path = data_config.dataset_path.cpc
            self.interpolated_cpc_path = data_config.dataset_path.interpolated_cpc
            self.interpolated_cpc_gauge_density_path = data_config.dataset_path.interpolated_cpc_gauge
        else:
            yprint('Computing CPC on the fly')
            self.cpc_aggregate_path = data_config.dataset_path.cpc
            self.cpc_gauge_density_path = data_config.dataset_path.cpc_gauge

        # Loading dates
        if not self.use_precomputed_mrms:
            files = os.listdir(self.mrms_path)
            precip_files = natsorted([f for f in files if '.nc' in f])
            # Regular expression to match the date part in the filenames
            date_pattern = re.compile(r'.*_(\d{8})-\d{6}\.nc')
            # Extract dates from the filenames
            dates = []
            for filename in precip_files:
                match = date_pattern.match(filename)
                if match:
                    # The first group captures the date
                    date = match.group(1)
                    if date not in dates:
                        dates.append(date)
            self.precip_dates = dates  # list of dates where MRMS data is available
            print('Loaded MRMS dataset containing {} combined daily aggregates'.format(len(self.precip_dates)))
        else:  # precomputed
            files = os.listdir(self.mrms_path)
            precip_files = natsorted([f for f in files if '.nc' in f])
            # files are named as mrms_daily_20150506.nc
            date_pattern = re.compile(r'mrms_daily_(\d{8})\.nc')
            dates = []
            for filename in precip_files:
                match = date_pattern.match(filename)
                if match:
                    date = match.group(1)
                    if date not in dates:
                        dates.append(date)
            self.precip_dates = dates
            # remove the date where MRMS switched to multisensor
            # date_to_remove = data_config.mrms_switched_to_multisensor_on
            dates_to_remove = data_config.invalid_daily_aggregates
            self.precip_dates = [d for d in self.precip_dates if d not in dates_to_remove]
            self.hourly_aggregate_def = data_config.hourly_aggregate_def
            assert self.hourly_aggregate_def in ['starting', 'ending'], \
                f'Invalid hourly aggregate definition: {self.hourly_aggregate_def}'
            print('Loaded MRMS dataset containing {} daily aggregates'.format(len(self.precip_dates)))

        # define normalization functions
        self.precip_norm_func = data_config.precip_norm_func
        self.normalize_precip, self.inverse_normalize_precip = self.define_precip_normalization_func(self.precip_norm_func)
        # define constants
        # self.xmin, self.xmax = 0, 2326.7  # empirically acquired, raw min/max pixel values from dataset
        # self.theta = 0.17  # following https://ieeexplore.ieee.org/document/9246532
        self.xmax = float(data_config.precip_rescale_c)
        self.eps = 0.00001  # for numerical precision

        # era5 normalization constants
        self.surf_temp_min = 240.
        self.surf_temp_max = 320.
        self.elev_c = 30000
        self.vflux_c = 800
        self.wind_c = 50

        # define boundaries
        self.mrms_bounds = data_config.geo_bounds.mrms
        self.era5_bounds = data_config.geo_bounds.era5_conus
        self.loc_bounds = self.define_location_bounnds_via_era5(r=self.img_size // 2)
        self.skip_invalid_regions = data_config.skip_invalid_regions

        # gauge density data
        self.use_density = data_config.condition_mode == 6
        self.density_c = 20.  # cpc gauge density

        self.cmaps = data_config.cmaps

        # historical mode
        self.historical_mode = data_config.historical_mode

    def __len__(self):
        return len(self.precip_dates)

    """Boundary related methods"""

    @deprecated
    def define_location_bounnds_via_mrms(self, r):
        # self.parse_date(1) # read arbitrary file
        if self.use_precomputed_mrms:
            precip = self.read_precomputed_mrms_daily_aggregate(self.precip_dates[1])
        else:
            precip = self.read_mrms_daily_aggregate(self.precip_dates[1])
        # precip = self.correct_boundaries(precip) # FIXME this used to be enabled
        # batch = self.read_era5_files()
        # era5 = batch['surf_temp']

        lon_min_nearest = precip.lon.sel(lon=self.mrms_bounds.lon_min, method='nearest').values
        lon_max_nearest = precip.lon.sel(lon=self.mrms_bounds.lon_max, method='nearest').values
        lat_min_nearest = precip.lat.sel(lat=self.mrms_bounds.lat_min, method='nearest').values
        lat_max_nearest = precip.lat.sel(lat=self.mrms_bounds.lat_max, method='nearest').values

        x_min, y_min = np.where(precip.lon == lon_min_nearest)[0][0], np.where(precip.lat == lat_min_nearest)[0][0]
        x_max, y_max = np.where(precip.lon == lon_max_nearest)[0][0], np.where(precip.lat == lat_max_nearest)[0][0]
        lon_min, lon_max = precip.lon.isel(lon=x_min + r).values, precip.lon.isel(lon=x_max - r - 1).values
        lat_min, lat_max = precip.lat.isel(lat=y_min + r).values, precip.lat.isel(lat=y_max - r - 1).values
        return {'lon_min': lon_min, 'lon_max': lon_max, 'lat_min': lat_min, 'lat_max': lat_max}

    def define_location_bounnds_via_era5(self, r):
        if self.use_precomputed_mrms:
            precip = self.read_precomputed_mrms_daily_aggregate(self.precip_dates[1])
        else:
            precip = self.read_mrms_daily_aggregate(self.precip_dates[1])

        lon_min_nearest = precip.lon.sel(lon=self.era5_bounds.lon_min, method='nearest').values
        lon_max_nearest = precip.lon.sel(lon=self.era5_bounds.lon_max, method='nearest').values
        lat_min_nearest = precip.lat.sel(lat=self.era5_bounds.lat_min, method='nearest').values
        lat_max_nearest = precip.lat.sel(lat=self.era5_bounds.lat_max, method='nearest').values

        x_min, y_min = np.where(precip.lon == lon_min_nearest)[0][0], np.where(precip.lat == lat_min_nearest)[0][0]
        x_max, y_max = np.where(precip.lon == lon_max_nearest)[0][0], np.where(precip.lat == lat_max_nearest)[0][0]
        lon_min, lon_max = precip.lon.isel(lon=x_min + r).values, precip.lon.isel(lon=x_max - r - 1).values
        lat_min, lat_max = precip.lat.isel(lat=y_min + r).values, precip.lat.isel(lat=y_max - r - 1).values
        return {'lon_min': lon_min, 'lon_max': lon_max, 'lat_min': lat_min, 'lat_max': lat_max}

    def check_location_bounds(self, lon, lat):
        return self.loc_bounds['lon_min'] < lon < self.loc_bounds['lon_max'] and \
            self.loc_bounds['lat_max'] < lat < self.loc_bounds['lat_min']

    """data loading methods"""
    # @profile  # Decorator from line_profiler
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
            # print('loaded')
            ds, valid_mask = self.merge_all_dataarrays(cpc_lr, mrms, era5, density, date_=date_)
        else:
            ds, valid_mask = self.merge_all_dataarrays(cpc_lr, mrms, era5, date_=date_)
        # print('merged')
        location = self.select_location(cpc_lr, valid_mask)
        batch, batch_coord = self.select_crop(ds, location)
        # print('cropped')
        batch = self.correct_for_nan(batch)
        batch = self.normalize_precip(batch)
        batch = self.normalize_era5(batch)
        batch = self.normalize_density(batch)  # if needed
        batch = self.cvt_to_tensor(batch)
        # check dimension of batch
        for k, v in batch.items():
            if v.shape != (1, self.img_size, self.img_size):
                raise ValueError(f'Invalid shape {v.shape} for {k} for date {date_} at location {location}')
        batch_coord['date'] = date_
        # print('|', end="")
        return batch, batch_coord

    def get_arbitrary_item(self, date_, lon, lat):
        if self.historical_mode:
            return self.get_arbitrary_item_historical(date_, lon, lat)

        location = {'lon': lon, 'lat': lat}
        if self.check_location_bounds(lon, lat) is False:
            raise ValueError('Location ({}, {}) is out of bounds.'.format(lon, lat))
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
            density = None
            ds, valid_mask = self.merge_all_dataarrays(cpc_lr, mrms, era5, date_=date_)
        batch, batch_coord = self.select_crop(ds, location)
        xr_batch_low_res = self.select_crop_via_coords(cpc_lr, mrms, density, era5, batch_coord)
        batch = self.correct_for_nan(batch)
        batch = self.normalize_precip(batch)
        batch = self.normalize_era5(batch)
        batch = self.normalize_density(batch)
        batch = self.cvt_to_tensor(batch)
        return batch, batch_coord, xr_batch_low_res, valid_mask

    def get_arbitrary_item_historical(self, date_, lon, lat):
        # MRMS data is not available for historical mode
        print('Historical mode. Reading cpc from .gz or .nc. MRMS data is not available.')
        location = {'lon': lon, 'lat': lat}
        if self.check_location_bounds(lon, lat) is False:
            raise ValueError('Location ({}, {}) is out of bounds.'.format(lon, lat))
        if int(date_[:4]) >= 1979:
            cpc_lr = self.read_cpc_daily_aggregate(date_)
        else:
            cpc_lr = self.read_historical_cpc_daily_aggregate(date_)

        if self.use_precomputed_mrms:
            mrms = self.read_precomputed_mrms_daily_aggregate('20200101')
        else:
            mrms = self.read_mrms_daily_aggregate('20200101')

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
            density = None
            ds, valid_mask = self.merge_all_dataarrays(cpc_lr, mrms, era5, date_=date_)
        batch, batch_coord = self.select_crop(ds, location)
        xr_batch_low_res = self.select_crop_via_coords(cpc_lr, mrms, density, era5, batch_coord)
        batch = self.correct_for_nan(batch)
        batch = self.normalize_precip(batch)
        batch = self.normalize_era5(batch)
        batch = self.normalize_density(batch)
        batch = self.cvt_to_tensor(batch)
        return batch, batch_coord, xr_batch_low_res, valid_mask

    def read_cpc_daily_aggregate(self, date_: str) -> xr.DataArray:
        """
        Reads daily aggregate precipitation data from the CPC dataset in the CONUS region.
        :param date_: date in the format 'YYYYMMDD'
        """
        cpc_year = xr.open_dataarray(p.join(self.cpc_aggregate_path, f'precip.{date_[:4]}.nc'))
        lon_max, lon_min, lat_max, lat_min = self.mrms_bounds.lon_max, self.mrms_bounds.lon_min, \
            self.mrms_bounds.lat_max, self.mrms_bounds.lat_min
        cpc_day_conus = cpc_year.sel(time=date_, lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        return cpc_day_conus

    def read_precomputed_cpc_daily_aggregate(self, date_: str) -> xr.DataArray:
        cpc_day_conus = xr.open_dataarray(p.join(self.interpolated_cpc_path, f'cpc_interp_{date_}.nc'))
        return cpc_day_conus

    @staticmethod
    def find_mrms_param_name(dataset: xr.Dataset) -> str:
        """
        Depending on the date, MRMS parameters have different names. This function tries to identify the parameter
        """
        # Define possible parameter name patterns or specific names
        possible_names = ['param9.6.209', 'param37.6.209']
        for name in possible_names:
            if name in dataset.data_vars:
                return name
        raise ValueError("Expected MRMS parameter not found in the dataset")

    def read_mrms_daily_aggregate(self, date_: str) -> xr.DataArray:
        assert self.hourly_aggregate_def == 'starting', 'Only starting definition is supported for now.'
        # 12 Zulu the day before to 12 Zulu current day
        file_paths = generate_mrms_dailyagg_12z(self.mrms_path, date_)
        # Now, read the datasets using xarray
        mrms_day = xr.open_mfdataset(paths=file_paths, combine='by_coords', parallel=False)
        param_name = self.find_mrms_param_name(mrms_day)
        mrms_day = mrms_day[param_name]
        mrms_day = mrms_day.isel(alt=0)
        mrms_day = mrms_day.sum(dim='time', min_count=23)
        return mrms_day

    def read_precomputed_mrms_daily_aggregate(self, date_: str) -> xr.DataArray:
        mrms_day = xr.open_dataarray(p.join(self.mrms_path, f'mrms_daily_{date_}.nc'))
        return mrms_day

    def truncate_dataarray(self, da:xr.DataArray):
        lon_min, lon_max = self.era5_bounds.lon_min, self.era5_bounds.lon_max
        lat_min, lat_max = self.era5_bounds.lat_min, self.era5_bounds.lat_max
        truncated = da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        if truncated.lat.size == 0:
            truncated = da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))
        return truncated

    def read_cpc_gauge_data(self, date_: str):
        # cpc_day_ds = read_cpc_file(date_)
        data_dir = p.join(self.cpc_gauge_density_path, date_[:4])
        filename = p.join(data_dir, [f for f in os.listdir(data_dir) if date_ in f][0])
        if self.historical_mode:
            cpc_day_ds = read_cpc_gz_file_from(filename)
        else:
            cpc_day_ds = read_cpc_file_from(filename)
        assert cpc_day_ds is not None, f'No gauge data found for date {date_}'
        cpc_gauge_density = cpc_day_ds['station_number']
        lon_min, lon_max = self.era5_bounds.lon_min, self.era5_bounds.lon_max
        lat_min, lat_max = self.era5_bounds.lat_min, self.era5_bounds.lat_max
        return cpc_gauge_density.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))

    def read_precomputed_cpc_gauge_data(self, date_: str):
        cpc_gauge_density = xr.open_dataarray(p.join(self.interpolated_cpc_gauge_density_path, f'cpc_density_interp_{date_}.nc'))
        return cpc_gauge_density

    def read_historical_cpc_daily_aggregate(self, date_: str) -> xr.DataArray:
        print()
        data_dir = p.join(self.cpc_gauge_density_path, date_[:4])
        filename = p.join(data_dir, [f for f in os.listdir(data_dir) if date_ in f][0])
        cpc_day_ds = read_cpc_gz_file_from(filename)
        assert cpc_day_ds is not None, f'No gauge data found for date {date_}'
        cpc_gauge_density = cpc_day_ds['rainfall'] / 10.  # convert to mm
        lon_min, lon_max = self.era5_bounds.lon_min, self.era5_bounds.lon_max
        lat_min, lat_max = self.era5_bounds.lat_min, self.era5_bounds.lat_max
        return cpc_gauge_density.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))

    # def read_cpc_conus_precip(self, date_: str):
    #     """
    #     Read from real time (.RT) file downloaded from CPC FPT directory.
    #     * Requires downloading the file from CPC FPT directory
    #     * CONUS only, .25 degree resolution
    #     """
    #     # cpc_day_ds = read_cpc_file(date_)
    #     data_dir = p.join(self.cpc_gauge_density_path, date_[:4])
    #     filename = p.join(data_dir, [f for f in os.listdir(data_dir) if date_ in f][0])
    #     cpc_day_ds = read_cpc_file_from(filename)
    #     assert cpc_day_ds is not None, f'No gauge data found for date {date_}'
    #     cpc_gauge_density = cpc_day_ds['rainfall']
    #     lon_min, lon_max = self.era5_bounds.lon_min, self.era5_bounds.lon_max
    #     lat_min, lat_max = self.era5_bounds.lat_min, self.era5_bounds.lat_max
    #     return cpc_gauge_density.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))

    @staticmethod
    def merge_precip_dataarrays(cpc: xr.DataArray, mrms: xr.DataArray) -> xr.Dataset:
        """
        Merges the daily aggregate precipitation data from CPC and MRMS datasets.
        * Defines mutually valid region (where both datasets have valid data)
        * Interpolates CPC data to MRMS grid
        * Masks invalid data
        """
        cpc_interp = cpc.interp(lon=mrms.lon, lat=mrms.lat, method='linear')
        valid_mask = np.logical_and(np.isfinite(mrms), np.isfinite(cpc_interp))
        ds = xr.Dataset({'mrms': mrms, 'cpc': cpc_interp})
        ds['mrms'] = ds['mrms'].where(valid_mask)
        ds['cpc'] = ds['cpc'].where(valid_mask)
        return ds

    # @profile  # Decorator from line_profiler
    def merge_all_dataarrays(self, cpc: xr.DataArray, mrms: xr.DataArray, era5: dict, density: xr.DataArray = None,
                             date_: str = None) -> Tuple[xr.Dataset, xr.DataArray]:
        """
        Merges the daily aggregate precipitation data from CPC and MRMS datasets.
        * Defines mutually valid region (where both precip products have valid data)
        * Interpolates CPC and ERA5 data to MRMS grid
        * Masks invalid data only on two precip products
        """
        # make a copy of era5
        era5 = era5.copy()
        if self.use_precomputed_cpc:
            cpc_interp = self.read_precomputed_cpc_daily_aggregate(date_)
        else:
            cpc_interp = cpc.interp(lon=mrms.lon, lat=mrms.lat, method='linear')
        # valid_mask = np.logical_and(np.isfinite(mrms), np.isfinite(cpc_interp)) # slow
        valid_mask = mrms.notnull() & cpc_interp.notnull() # optimized
        if density is not None:
            if self.use_precomputed_cpc:
                density_interp = self.read_precomputed_cpc_gauge_data(date_)
            else:
                density_interp = density.interp(lon=mrms.lon, lat=mrms.lat, method='linear')
            valid_mask = np.logical_and(valid_mask, np.isfinite(density_interp))
        if self.use_precomputed_era5:
            era5_merged = era5
        else:
            era5_merged = xr.Dataset(era5)
            era5_merged = era5_merged.interp(lon=mrms.lon, lat=mrms.lat, method='linear')

        if density is None:
            ds = xr.merge([xr.Dataset({'mrms': mrms, 'cpc': cpc_interp}), xr.Dataset(era5)], compat='override')
        else:
            # ds = xr.merge([xr.Dataset({'mrms': mrms, 'cpc': cpc_interp, 'density': density_interp}), xr.Dataset(era5)], compat='override')
            ds = xr.Dataset({'mrms': mrms, 'cpc': cpc_interp, 'density': density_interp, **era5_merged})
        ds['mrms'] = ds['mrms'].where(valid_mask)
        ds['cpc'] = ds['cpc'].where(valid_mask)
        if density is not None:
            ds['density'] = ds['density'].where(valid_mask)
        return ds, valid_mask

    def read_era5_files(self, date_: str) -> dict:
        # 12 - 12 Zulu
        year_ = date_[:4]
        batch = {}
        for k, v in self.era5_filenames.items():
            f = xr.open_dataarray(p.join(self.era5_path, v.format(year_)))
            if k != 'elevation':
                if self.hourly_aggregate_def == 'starting':
                    start_time = pd.to_datetime(date_) - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
                    end_time = pd.to_datetime(date_) + pd.Timedelta(hours=11)
                elif self.hourly_aggregate_def == 'ending':
                    start_time = pd.to_datetime(date_) - pd.Timedelta(days=1) + pd.Timedelta(hours=13)
                    end_time = pd.to_datetime(date_) + pd.Timedelta(hours=12)
                f = f.sel(time=slice(start_time, end_time))
                f = f.sum(dim='time', min_count=23) / 24.  # compute daily AVERAGE
            else:  # elevation, static
                f = f[0, :, :]
            f = self.cvt_180_to_360(f)
            f = self.correct_axis(f)
            f = f.assign_coords(time=pd.to_datetime(date_))  # TODO verify this
            batch[k] = f
        return batch

    def read_precomputed_era5_files(self, date_: str) -> xr.Dataset:
        era5 = xr.open_dataset(p.join(self.era5_path, self.era5_filenames.format(date_)))
        return era5


    """Data processing methods"""

    def correct_boundaries(self, x):
        """
        Corrects the boundaries of the image.
        :param x: xarray
        :return: corrected image
        """
        x = x.sel(lat=slice(self.era5_bounds.lat_min, self.era5_bounds.lat_max),
                  lon=slice(self.era5_bounds.lon_min, self.era5_bounds.lon_max))
        return x

    def build_coarse_cdf(self, precip):
        """
        Builds a 1-d coarse CDF of the precipitation image.
        """
        precip_coarsened = precip.coarsen(lon=self.img_size, lat=self.img_size, boundary='trim').mean()
        precip_coarsened_bin = precip_coarsened > 0.2
        cdf = np.cumsum(precip_coarsened_bin.to_numpy().ravel())
        return cdf

    @staticmethod
    def sample_from_cdf(cdf):
        random_index = np.searchsorted(cdf, np.random.random() * cdf[-1])
        return random_index

    def select_location(self, lr_precip: xr.DataArray, valid_mask: xr.DataArray) -> dict:
        precip_coarsened_bin = lr_precip > 0.2

        width = precip_coarsened_bin.shape[-1]
        cdf = np.cumsum(precip_coarsened_bin.to_numpy().ravel())
        # try:
        attempts = 5
        for _ in range(attempts):
            idx = self.sample_from_cdf(cdf)
            y, x = divmod(idx, width)
            # y, x = y * self.img_size, x * self.img_size
            lon = lr_precip.lon.isel(lon=x).values
            lat = lr_precip.lat.isel(lat=y).values
            if self.check_location_bounds(lon, lat):
                # take a slice in valid mask
                lon_idx = np.abs(valid_mask.coords['lon'] - lon).argmin().values.item()
                lat_idx = np.abs(valid_mask.coords['lat'] - lat).argmin().values.item()
                # Calculate half sizes for slicing
                half_size_y, half_size_x = self.img_size // 2, self.img_size // 2
                # Calculate start and end indices for both dimensions
                start_lat_idx = lat_idx - half_size_y
                end_lat_idx = lat_idx + half_size_y
                start_lon_idx = lon_idx - half_size_x
                end_lon_idx = lon_idx + half_size_x
                proposed_crop = valid_mask.isel(lat=slice(start_lat_idx, end_lat_idx), lon=slice(start_lon_idx, end_lon_idx))
                # check if the proposed crop is valid
                if proposed_crop.all():
                    return {'lon': lon, 'lat': lat}
        # unable to find a crop with rainfall, so just return a random crop
        lon = np.random.uniform(self.loc_bounds['lon_min']+2, self.loc_bounds['lon_max']-2)
        lat = np.random.uniform(self.loc_bounds['lat_max']+2, self.loc_bounds['lat_min']-2)
        return {'lon': lon, 'lat': lat}

    @staticmethod
    def cvt_360_to_180(df):
        df.coords['lon'] = (df.coords['lon'] + 180) % 360 - 180
        df = df.sortby(df.lon)
        return df

    @staticmethod
    def cvt_180_to_360(df):
        df.coords['longitude'] = (df.coords['longitude'] % 360 + 360) % 360
        df = df.sortby(df.longitude)
        return df

    @staticmethod
    def correct_axis(x):
        return x.rename({'longitude': 'lon', 'latitude': 'lat'})

    def select_crop(self, ds: xr.Dataset, location: dict) -> tuple[dict[str | Hashable, ndarray], dict[str, ndarray]]:
        """
        Returns a set of crops where ERA5 and GT precip have the same FoV. Raw ERA5 have lower resolution than GT precip.
        """
        lon, lat = location['lon'], location['lat']
        # Find the closest index in the DataArray to the given lon and lat
        lon_idx = np.abs(ds.coords['lon'] - lon).argmin().values.item()
        lat_idx = np.abs(ds.coords['lat'] - lat).argmin().values.item()
        # Calculate half sizes for slicing
        half_size_y, half_size_x = self.img_size // 2, self.img_size // 2
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

    @staticmethod
    def select_crop_via_coords(cpc: xr.DataArray, mrms: xr.DataArray, density: xr.DataArray,
                               era5: dict, crop_coords: dict):
        """
        Returns a set of crops where each input have the original (possible low) resoution.
        """
        # Extract the coordinates from the crop_coords dictionary
        lon_min = crop_coords['precip_lon_min']
        lon_max = crop_coords['precip_lon_max']
        lat_min = crop_coords['precip_lat_min']
        lat_max = crop_coords['precip_lat_max']
        # Select the sub-arrays using the coordinates
        cpc_crop = cpc.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        if cpc_crop.lat.size == 0:
            cpc_crop = cpc.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))
        mrms_crop = mrms.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        if density is not None:
            density_crop = density.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))  # slice for lat somehow needs to be reversed
        else:
            density_crop = None
        # For era5, since it's a dictionary of DataArrays, we need to do this for each DataArray
        era5_crop = {k: v.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max)) for k, v in era5.items()}
        # return dictionary
        return {'precip_lr': cpc_crop, 'precip_gt': mrms_crop, 'density': density_crop, **era5_crop}


    @staticmethod
    def correct_for_nan(batch) -> dict:
        for k, arr in batch.items():
            arr[np.isnan(arr)] = -1
            # arr[arr < 0.] = 0.
            batch[k] = arr
        return batch

    def define_precip_normalization_func(self, func_name: str):
        if func_name == 'linear':
            print('Using linear normalization for precipitation')
            return self.normalize_precip_linear, self.inverse_normalize_precip_linear
        elif func_name == 'log':
            print('Using log normalization for precipitation')
            return self.normalize_precip_log, self.inverse_normalize_precip_log
        elif func_name == 'log_plus_one':
            print('Using log + 1 normalization for precipitation')
            return self.normalize_precip_log_plus_one, self.inverse_normalize_precip_log_plus_one
        else:
            raise ValueError('Invalid normalization function name')

    def normalize_precip_linear(self, batch):
        """
        Normalizes input data. Maps valid pixel input [0, xmax] to [0, 1].
        Maps invalid pixels to -1.
        """
        for k, arr in batch.items():
            if 'precip' in k:
                precip = batch[k]
                valid_region = precip != -1
                precip = precip / self.xmax
                precip[~valid_region] = -1
                batch[k] = precip
        return batch

    def normalize_precip_log(self, batch):
        """
        Normalizes input data by taking log. Maps valid pixel input [xmin, xmax] to [theta, 1].
        Maps invalid pixels to -1.
        ISSUE: Makes the model very hard at predicting zero-rainfall regions.
        :param img:
        :return:
        """
        for k, arr in batch.items():
            if 'precip' in k:
                precip = batch[k]
                # scale by taking log
                valid_region = precip != -1
                scaled = np.zeros_like(precip)
                # TODO verify this, used to get rid of invalid value encountered in log
                precip[valid_region] = np.where(precip[valid_region] < 0, self.eps, precip[valid_region])
                scaled[valid_region] = np.log(precip[valid_region] + self.eps)
                # normalized = scaled
                normalized = scaled / 5.  # scale to 0 to 1 TODO not scientific
                # normalized[normalized < self.theta] = 0  # clip lower end at theta
                normalized[~valid_region] = -1
                batch[k] = normalized
        return batch

    def normalize_precip_log_plus_one(self, batch):
        for k, arr in batch.items():
            if 'precip' in k:
                precip = batch[k]
                # scale by taking log
                valid_region = precip != -1
                scaled = np.zeros_like(precip)
                scaled[valid_region] = np.log(precip[valid_region] + 1)
                normalized = scaled / 5.  # scale to 0 to 1 TODO not scientific
                normalized[~valid_region] = -1
                batch[k] = normalized
        return batch

    def normalize_era5(self, batch):
        batch['surf_temp'] = (batch['surf_temp'] - self.surf_temp_min) / (self.surf_temp_max - self.surf_temp_min)
        batch['elevation'] = batch['elevation'] / self.elev_c
        batch['vflux_e'] = batch['vflux_e'] / self.vflux_c
        batch['vflux_n'] = batch['vflux_n'] / self.vflux_c
        batch['wind_u'] = batch['wind_u'] / self.wind_c
        batch['wind_v'] = batch['wind_v'] / self.wind_c
        return batch

    def normalize_density(self, batch):
        if self.use_density:
            batch['density'] = batch['density'] / self.density_c
            return batch
        else:
            return batch

    def inverse_normalize_precip_linear(self, precip_t):
        """
        Inverse of input data normalization. Maps normalized rainfall range [0, 1] back to [0, xmax].
        Invalid pixels (-1) are mapped to -1.
        :param precip_t:
        :return:
        """
        # identify regions
        invalid_region = precip_t == -1  # originally np.nan
        rainfall_region = precip_t > 0  # positive rainfall
        # initialize array to hold the inverse values
        inversed = torch.zeros_like(precip_t)
        # reverse the normalization and scaling for valid regions
        inversed[rainfall_region] = precip_t[rainfall_region] * self.xmax
        inversed[invalid_region] = -1
        return inversed

    def inverse_normalize_precip_log_plus_one(self, precip_t):
        """
        Inverse of input data normalization. Maps normalized rainfall range [0, 1] back to [0, xmax].
        Invalid pixels (-1) are mapped to -1.
        :param precip_t:
        :return:
        """
        # identify regions
        invalid_region = precip_t == -1
        rainfall_region = precip_t > 0
        # initialize array to hold the inverse values
        inversed = torch.zeros_like(precip_t)
        # reverse the normalization and scaling for valid regions
        inversed[rainfall_region] = torch.exp(precip_t[rainfall_region] * 5) - 1
        inversed[invalid_region] = -1
        return inversed

    def inverse_normalize_precip_log(self, precip_t):
        """
        Inverse of input data normalization. Maps normalized rainfall range [theta, 1] back to [xmin, xmax].
        Dry regions (0) are mapped to 0. Invalid pixels (-1) are mapped to -1.
        :param precip_t:
        :return:
        """
        # identify regions
        invalid_region = precip_t == -1  # originally np.nan
        dry_region = precip_t == 0  # no rainfall
        rainfall_region = precip_t > 0  # positive rainfall
        # initialize array to hold the inverse values
        inversed = torch.zeros_like(precip_t)
        # reverse the normalization and scaling for valid regions
        inversed[rainfall_region] = torch.exp(precip_t[rainfall_region] * 5) - self.eps  # FIXME: magic number 5
        inversed[dry_region] = 0
        inversed[invalid_region] = -1
        return inversed

    def inverse_normalize_batch(self, batch):
        # FIXME: parameter object is also modified
        for k, arr in batch.items():
            if 'precip' in k:
                batch[k] = self.inverse_normalize_precip(arr)
        batch['surf_temp'] = batch['surf_temp'] * (self.surf_temp_max - self.surf_temp_min) + self.surf_temp_min
        batch['elevation'] = batch['elevation'] * self.elev_c
        batch['vflux_e'] = batch['vflux_e'] * self.vflux_c
        batch['vflux_n'] = batch['vflux_n'] * self.vflux_c
        batch['wind_u'] = batch['wind_u'] * self.wind_c
        batch['wind_v'] = batch['wind_v'] * self.wind_c
        if self.use_density:
            batch['density'] = batch['density'] * self.density_c
        return batch

    @staticmethod
    def cvt_to_tensor(batch) -> dict:
        for k, arr in batch.items():
            # if arr is numpy array
            if isinstance(arr, np.ndarray):
                t = torch.from_numpy(arr).float()
            # if arr is xarray dataarray
            elif isinstance(arr, xr.DataArray):
                t = torch.from_numpy(arr.to_numpy()).float()
            else:
                raise ValueError('Invalid type for arr')
            batch[k] = t.unsqueeze(dim=0)
        return batch

    @staticmethod
    def cvt_batch_to_xarray(batch, batch_coords) -> dict:
        xarray_batch = {}
        for k, v in batch.items():
            # assuming same lon lat for precip and era5
            lon_min = batch_coords['precip_lon_min']
            lon_max = batch_coords['precip_lon_max']
            lat_min = batch_coords['precip_lat_min']
            lat_max = batch_coords['precip_lat_max']
            v = v.squeeze(dim=0).detach().cpu().numpy()
            len_lat, len_lon = v.shape[-2], v.shape[-1]
            lon = np.linspace(lon_min, lon_max, len_lon)
            lat = np.linspace(lat_min, lat_max, len_lat)
            da = xr.DataArray(v, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name=k)
            xarray_batch[k] = da
        return xarray_batch

    @staticmethod
    def plot_composite_xarray_batch_conus(ds: xr.Dataset, save_dir: str = None, rainfall_vis_max: float = None):
        """
        Plots the composite xarray dataset containing multiple variables.
        Expects matching dims and coords for all variables.
        """

        if rainfall_vis_max is None:
            rainfall_vis_max = ds['mrms'].max().item()
        for var_name in ds.data_vars:
            if 'precip' in var_name or 'cpc' in var_name or 'mrms' in var_name:
                ds[var_name].plot(vmin=0, vmax=rainfall_vis_max)
                plt.title(var_name)
            # composite
            elif var_name == 'wind_u':
                composite_var_name = 'wind'
                norm_val = np.sqrt(ds['wind_u'] ** 2 + ds['wind_v'] ** 2)
                norm = xr.DataArray(norm_val, coords=ds['wind_u'].coords, dims=ds['wind_u'].dims)
                fig, ax = plt.subplots()
                norm.plot.contourf(ax=ax, cmap='jet', levels=10)
                temp_ds = xr.Dataset({'e': ds['wind_u'], 'n': ds['wind_v']})
                vector_field = temp_ds.coarsen(lon=120, lat=120, boundary='trim').mean()
                vector_field.plot.quiver(x='lon', y='lat', u='e', v='n', ax=ax)
                plt.title(composite_var_name)
            elif var_name == 'vflux_e':
                composite_var_name = 'vflux'
                norm_val = np.sqrt(ds['vflux_e'] ** 2 + ds['vflux_n'] ** 2)
                norm = xr.DataArray(norm_val, coords=ds['vflux_e'].coords, dims=ds['vflux_e'].dims)
                fig, ax = plt.subplots()
                norm.plot.contourf(ax=ax, cmap='jet', levels=10)
                temp_ds = xr.Dataset({'e': ds['vflux_e'], 'n': ds['vflux_n']})
                vector_field = temp_ds.coarsen(lon=120, lat=120, boundary='trim').mean()
                vector_field.plot.quiver(x='lon', y='lat', u='e', v='n', ax=ax)
                plt.title(composite_var_name)
            elif var_name in ['surf_temp', 'elevation', 'density']:
                ds[var_name].plot()
                plt.title(var_name)

            if save_dir is not None:
                plt.savefig(p.join(save_dir, var_name + '.png'))
                print('Saved composite plots to {}'.format(save_dir))
            else:
                plt.show()
            plt.close()

    # @staticmethod
    def plot_composite_xarray_batch(self, batch, rainfall_vis_max: float = None, save_dir: str = None,
                                    use_upsampled: bool = True, save_netcdf: bool = False):
        if use_upsampled:
            r = 24
        else:
            r = 1  # coarsening factor for vector field
        if rainfall_vis_max is None:
            rainfall_vis_max = max([v.max() for k, v in batch.items() if 'precip' in k])
        for k, v in batch.items():
            if k in ['wind', 'vflux']:
                plt.rcParams.update({'font.size': 16})
                norm_val = np.sqrt(v['e'] ** 2 + v['n'] ** 2)
                # norm = xr.DataArray(norm_val, coords=v.coords, dims=v.dims)
                norm = xr.DataArray(norm_val, coords=norm_val.coords, dims=norm_val.dims)
                fig, ax = plt.subplots()
                norm.plot.contourf(ax=ax, cmap='jet', levels=10)
                vector_field = v.coarsen(lon=r, lat=r, boundary='trim').mean()
                vector_field.plot.quiver(x='lon', y='lat', u='e', v='n', ax=ax)
            elif 'precip' in k:
                plt.rcParams.update({'font.size': 14})
                v.plot(vmax=rainfall_vis_max, vmin=0)
            else:
                plt.rcParams.update({'font.size': 16})
                cmap_ = self.cmaps[k] if k in self.cmaps else None
                vmin_ = 0 if k == 'elevation' else None
                v.plot(cmap=cmap_, vmin=vmin_)
            plt.title(k)
            if save_dir is not None:
                k_ = k + '_lr' if not use_upsampled else k
                plt.savefig(p.join(save_dir, k_ + '.png'), dpi=600)
                # print('Saved composite plots to {}'.format(save_dir))
                if save_netcdf:
                    v.to_netcdf(p.join(save_dir, k_ + '.nc'))
            else:
                plt.show()
                if save_netcdf:
                    rprint('Cannot save netcdf without specifying save_dir')
            plt.close()


    @staticmethod
    def create_composite_xarray_batch(batch: dict) -> dict:
        """
        Creates composite plots of the era5 data. Requires all elements in batch to be in xarray format
        """
        # wind
        e = batch['wind_u']
        n = batch['wind_v']
        composite = xr.Dataset({'e': e, 'n': n})
        batch['wind'] = composite
        del batch['wind_u']
        del batch['wind_v']

        # wator vapor flux
        e = batch['vflux_e']
        n = batch['vflux_n']
        composite = xr.Dataset({'e': e, 'n': n})
        batch['vflux'] = composite
        del batch['vflux_e']
        del batch['vflux_n']

        return batch

    def plot_tensor_batch(self, batch: torch.tensor, batch_coords: dict, rainfall_vis_max: float = None,
                          save_dir: str = None, save_netcdf: bool = False) -> dict:
        """
        Plots the batch of tensors. Requires all elements in batch to be in tensor format.
        Returns xarray batch recaled to mm/day.
        NOTE: pointer to batch is also rescaled, event without return statement.
        """
        batch = self.inverse_normalize_batch(batch)
        xarray_batch = self.cvt_batch_to_xarray(batch, batch_coords)
        xarray_batch = self.create_composite_xarray_batch(xarray_batch)
        self.plot_composite_xarray_batch(xarray_batch, rainfall_vis_max=rainfall_vis_max, save_dir=save_dir,
                                         save_netcdf=save_netcdf)
        return xarray_batch


def debug_dataloader(cfg):
    """
    Checks run time
    """
    batch_size = 6 # 12
    cfg.data.num_workers = 6
    cfg.data.batch_size = batch_size
    # cfg.data.data_config.use_precomputed_cpc = True
    OmegaConf.set_struct(cfg, False)

    cfg.data.train_val_split = 0.8
    cfg.seed = 42
    train_loader, eval_loader, dateset = get_precip_era5_dataset(cfg, eval_num_worker=cfg.data.num_workers)
    train_iter = iter(train_loader)
    eval_iter = iter(eval_loader)

    progress = get_training_progressbar()
    total_ = 30
    with progress:
        task = progress.add_task(f"[Debug] Generating validation inputs", total=total_, start=True)
        start_time = time.monotonic()
        for step in range(total_):
            try:
                batch_dict, batch_coord = next(train_iter)
                # save batch_dict and batch_coord
                progress.update(task, advance=1)

            except RuntimeError:
                print("Timeout error occurred. Skipping this batch.")
                continue
        yprint('---------------------------------')
        stop_time = time.monotonic()
        yprint(f'Processing time = {dt.timedelta(seconds=stop_time - start_time)}')
    return


if __name__ == '__main__':
    with initialize(version_base=None, config_path="../../configs", job_name="evaluation"):
        config = compose(config_name="train")
    debug_dataloader(config)