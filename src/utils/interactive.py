"""
Util functions for interactive visualization (e.g., folium, ipyleaflet)
"""
__author__ = 'yuhao liu'

import os
import xarray as xr

def read_output_nc(file_path: str, var_name: str = 'precip_output') -> xr.Dataset:
    """
    Read NetCDF file (lon, lat) and return xarray DataArray with spatial dims and CRS set.
    Parameters:
    - file_path: str, path to the NetCDF file.
    - var_name: str, name of the variable to extract from the dataset.
    Returns:
    - da: xarray.DataArray with spatial dimensions and CRS set.
    """
    # open dataset
    assert os.path.isfile(file_path), f"File not found: {file_path}"
    ds  = xr.open_dataset(file_path)
    da  = ds[var_name]
    # make longitude −180…180
    if da.lon.max() > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby('lon')  # wrap & sort
    # ensure latitude is north-to-south
    if da.lat[0] < da.lat[-1]:
        da = da.sortby('lat', ascending=False)
    # tell rioxarray which dims are spatial and write CRS
    da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    da.rio.write_crs("EPSG:4326", inplace=True)
    return da
