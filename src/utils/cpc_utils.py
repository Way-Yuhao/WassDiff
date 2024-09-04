__author__ = 'yuhao liu'

import os
import os.path as p
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from ftplib import FTP
from rich.progress import Progress
from hydra import initialize, compose
import pandas as pd
import glob
import gzip


def download_cpc_data(year: str, out_dir: str):
    """
    Download CPC daily precip (CONUS, RT) data for a given year.
    """
    ftp_server = "ftp.cpc.ncep.noaa.gov"
    directory = f"/precip/CPC_UNI_PRCP/GAUGE_CONUS/RT/{year}/"
    # Local directory to save files
    os.makedirs(p.join(out_dir, year), exist_ok=True)
    # Connect to the FTP server
    with FTP(ftp_server) as ftp:
        ftp.login()  # Anonymous login
        ftp.cwd(directory)
        # List files in the directory
        files = ftp.nlst()
        # Create a progress bar
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Downloading year {year}", total=len(files))
            for filename in files:
                local_path = os.path.join(out_dir, year, filename)
                with open(local_path, 'wb') as file:
                    ftp.retrbinary(f'RETR {filename}', file.write)

                # Update the progress bar
                progress.update(task, advance=1)
    print("Download completed.")


def read_cpc_gz_file_from(filename: str):
    # Assuming the data is stored in a little-endian float format (4 bytes per value)
    # and the grid size is 300x120
    # Adjusted dimensions for the US grid

    # Open and read the gzipped file
    with gzip.open(filename, 'rb') as file:
        rain = np.frombuffer(file.read(300 * 120 * 4), dtype=np.float32).reshape((120, 300))
        stnm = np.frombuffer(file.read(300 * 120 * 4), dtype=np.float32).reshape((120, 300))

    rain = np.where(rain == -999, np.nan, rain)
    stnm = np.where(stnm == -999, np.nan, stnm)

    # Latitude and Longitude ranges for the US grid
    lat = np.linspace(20.125, 49.875, 120)
    lon = np.linspace(230.125, 304.875, 300)

    rain_da = xr.DataArray(rain, coords=[lat, lon], dims=["lat", "lon"], name="rainfall")
    stnm_da = xr.DataArray(stnm, coords=[lat, lon], dims=["lat", "lon"], name="station_number")
    ds = xr.Dataset({"rainfall": rain_da, "station_number": stnm_da})

    return ds

def read_cpc_file_from(filename: str):
    # Assuming the data is stored in a little-endian float format (4 bytes per value)
    # and the grid size is 300x120
    # Adjusted dimensions for the US grid
    rain = np.fromfile(filename, dtype=np.float32, count=300 * 120, offset=0).reshape((120, 300))
    stnm = np.fromfile(filename, dtype=np.float32, count=300 * 120, offset=300 * 120 * 4).reshape((120, 300))

    rain = np.where(rain == -999, np.nan, rain)
    stnm = np.where(stnm == -999, np.nan, stnm)

    # Latitude and Longitude ranges for the US grid
    lat = np.linspace(20.125, 49.875, 120)
    lon = np.linspace(230.125, 304.875, 300)

    rain_da = xr.DataArray(rain, coords=[lat, lon], dims=["lat", "lon"], name="rainfall")
    stnm_da = xr.DataArray(stnm, coords=[lat, lon], dims=["lat", "lon"], name="station_number")
    ds = xr.Dataset({"rainfall": rain_da, "station_number": stnm_da})
    return ds


def read_cpc_file(date_: str):
    with initialize(version_base=None, config_path="../configs"):
        # Compose the configuration using the config name and any overrides you wish to apply
        cfg = compose(config_name="downscale_cpc")
        data_dir = cfg.data.dataset_path.cpc_gauge
    year_ = date_[:4]
    data_dir = p.join(data_dir, date_[:4])
    # filename = p.join(data_dir, f'PRCP_CU_GAUGE_V1.0CONUS_0.25deg.lnx.{date_}.RT')
    filename = p.join(data_dir,[f for f in os.listdir(data_dir) if date_ in f][0])
    print(f"Reading data from {filename}") # TODO remove this
    ds = read_cpc_file_from(filename)
    return ds


def generate_mrms_dailyagg_12z(mrms_path, current_date):
    """
    Generate a list of file paths for MRMS 1-hourly precipitation data for a given date.
    The list contains all files from 12z of the previous day to 11z of the current day.
    """
    # Define the datetime range
    start_time = pd.to_datetime(current_date) - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
    end_time = pd.to_datetime(current_date) + pd.Timedelta(hours=11)

    # Generate a list of all hourly timestamps in the range
    time_range = pd.date_range(start=start_time, end=end_time, freq='H')

    # Generate file paths
    file_paths = []
    for timestamp in time_range:
        # Construct a search pattern that matches both filename conventions
        search_pattern = f"*_*_01H_*_{timestamp.strftime('%Y%m%d-%H%M%S')}.nc"
        full_search_path = os.path.join(mrms_path, search_pattern)

        # Use glob to find files that match the pattern
        matching_files = glob.glob(full_search_path)

        # Add all found files to the list (assuming we can handle multiple files for the same timestamp if they exist)
        file_paths.extend(matching_files)

    return file_paths


if __name__ == '__main__':
    # example_path = '/home/yl241/data/CLIMATE/CPC/examples/PRCP_CU_GAUGE_V1.0CONUS_0.25deg.lnx.20190929.RT'
    # read_cpc_file(example_path)
    # download_cpc_data(year='2015', out_dir='/home/yl241/data/CLIMATE/CPC/GAUGE_CONUS_RT')

    # read_cpc_file(date_='20150101')
    # example_path = '/home/yl241/data/CLIMATE/CPC/GAUGE_CONUS_RT/2019/PRCP_CU_GAUGE_V1.0CONUS_0.25deg.lnx.20191001.RT'
    example_path = '/home/yl241/Downloads/FTP/ftp.cpc.ncep.noaa.gov/precip/CPC_UNI_PRCP/GAUGE_CONUS/V1.0/2000/PRCP_CU_GAUGE_V1.0CONUS_0.25deg.lnx.20001226.gz'
    ds = read_cpc_gz_file_from(example_path)
    print(ds)
    ds['station_number'].plot()
    plt.show()

