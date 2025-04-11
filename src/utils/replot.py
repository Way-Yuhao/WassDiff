import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

def vis_sample(nc_file: str, ground_truth: str, density_file: str, save_dir: str, batch: str, index: int):
    """
    Visualize a sample from a NetCDF file using xarray, masking out regions where the
    interpolated density is NaN or zero. This ensures we preserve the resolution of
    the original precipitation grid by interpolating the lower-resolution density
    data to match the precipitation grid.

    :param nc_file: Path to the NetCDF (.nc) file containing the 'precip_output' variable.
    :param ground_truth: Path to a ground truth NetCDF file (currently unused in this example).
    :param density_file: Path to the NetCDF (.nc) file containing the 'density' variable (e.g. 'station_number').
    :param save_dir: Directory where the output plot will be saved.
    :param batch: Identifier (e.g., filename) used to derive the base name for saving the plot.
    :param index: Sample index used in the output filename.
    """
    #Open the NetCDF files
    ds = xr.open_dataset(nc_file)
    dp = xr.open_dataset(ground_truth)  # Not used here, but opened for completeness
    dd = xr.open_dataset(density_file)

    #Extract the precipitation and density DataArrays
    # precip_output = ds['precip_output']
    precip_output = dp['precip']
    # High-resolution precipitation
    density_coarse = dd['station_number']

    # Sort coordinates (lat/lon must be in ascending order for interpolation)
    precip_output = precip_output.sortby('lat').sortby('lon')
    density_coarse = density_coarse.sortby('lat').sortby('lon')

    # Interpolate the coarse density to match the precipitation grid
    density_interp = density_coarse.interp(
        lat=precip_output.lat,
        lon=precip_output.lon,
        method='linear'
    )

    precip_output = precip_output.clip(min=0)
    masked_output = precip_output.where((density_interp.notnull()))

    # Define color bar scale limits
    vmin_ = 0
    vmax_ = 150

    # Create a colormap that displays masked (NaN) values in white
    cmap = mpl.cm.get_cmap('viridis').copy()
    cmap.set_bad(color='white')  # Regions with masked values will be shown in white

    # Plot the masked DataArray
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = masked_output.plot(
        ax=ax,
        vmin=vmin_,
        vmax=vmax_,
        cmap=cmap,
        rasterized=True,
        cbar_kwargs={"label": ""}
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Save the figure
    basename = batch.replace('.pt', '')
    output_path = os.path.join(save_dir, f'{basename}_{index}.svg')
    plt.savefig(output_path, dpi=500, transparent=True)
    plt.close()
    print(f"Plot saved to {output_path}")


if __name__ == '__main__':
    nc_file_path = "/scratch/qdai/data/rainfall_plots_LiT/storm_bill_full_size/precip_output.nc"
    ground_truth_path = "/scratch/qdai/data/rainfall_plots_LiT/storm_bill_full_size/precip_lr_lr.nc"
    density_file_path = "/scratch/qdai/data/rainfall_plots_LiT/storm_bill_full_size/density_lr.nc"
    save_directory = "/scratch/qdai/data/rainfall_plots_LiT/storm_bill_full_size"

    batch_id = "precip.pt"
    sample_index = 0

    vis_sample(nc_file_path, ground_truth_path, density_file_path, save_directory, batch_id, sample_index)
