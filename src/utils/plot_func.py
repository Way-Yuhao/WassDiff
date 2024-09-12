from typing import Dict, List
import os
import os.path as p
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from src.utils.pysteps.spectral import rapsd


def plot_error_map(output, gt, save_dir, suffix=None):
    error = output - gt
    plt.imshow(error, vmin=-10, vmax=10, cmap='RdBu')
    cbar = plt.colorbar()
    cbar.set_label('Precipitation rate (mm/h)')
    plt.title('Error Map')
    if save_dir is not None:
        suffix = '' if suffix is None else '_' + suffix
        plt.savefig(p.join(save_dir, f'error_map{suffix}.png'))
    else:
        plt.show()
    plt.close()
    return


def plot_distribution(output, gt, save_dir, y_log_scale=True, suffix=None):
    # Calculate common x and y limits
    min_val = min(output.min(), gt.min())
    max_val = max(output.max(), gt.max())
    max_count = 30000  # for linear scale

    # Create a DataFrame with 'output' and 'gt' arrays
    df = pd.DataFrame({
        'Values': np.concatenate([output.flatten(), gt.flatten()]),
        'Type': ['Output'] * len(output.flatten()) + ['Ground Truth'] * len(gt.flatten())
    })

    # Plot the histogram
    sns.histplot(data=df, x='Values', hue='Type', bins=40, log_scale=(False, y_log_scale))
    plt.xlim([min_val, max_val])  # Set x limit
    if y_log_scale:
        plt.title('Distribution of precipitation rate (log scale)')
        suffix = '' if suffix is None else '_' + suffix
        fname = p.join(save_dir, f'distribution_log_scale{suffix}.png')
        plt.gca().set_ylim(bottom=1)  # Set y limit
    else:
        plt.ylim([0, max_count])  # Set y limit
        plt.title('Distribution of precipitation rate')
        suffix = '' if suffix is None else '_' + suffix
        fname = p.join(save_dir, f'distribution{suffix}.png')
    plt.xlabel('Precipitation rate (mm/h)')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(fname)
    else:
        plt.show()
    plt.close()
    return


def plot_qq(output, gt, save_dir, suffix=None):
    """
    Quantile-Quantile Plot
    https://stackoverflow.com/questions/46935289/quantile-quantile-plot-using-seaborn-and-scipy
    """
    # sns.set_theme(style="whitegrid")
    sns.despine()
    a = output.flatten()
    b = gt.flatten()
    # calculate quantiles
    percs = np.linspace(0, 100, 500)
    qn_a = np.percentile(a, percs)
    qn_b = np.percentile(b, percs)
    # plot dots
    # plt.plot(qn_a, qn_b, ls="", marker="o")
    sns.scatterplot(x=qn_a, y=qn_b, linewidth=0, color='xkcd:steel')
    # plot y = x ideal line
    x = np.linspace(np.min((qn_a.min(), qn_b.min())), np.max((qn_a.max(), qn_b.max())))
    sns.lineplot(x=x, y=x, color="k", ls="--")
    plt.title('Quantile-Quantile Plot')
    plt.xlabel('Output (mm/h)')
    plt.ylabel('Ground Truth (mm/h)')
    if save_dir is not None:
        suffix = '' if suffix is None else '_' + suffix
        plt.savefig(p.join(save_dir, f'qq_plot{suffix}.png'))
    else:
        plt.show()
    plt.close()
    return


def plot_qq_ensemble(batch, num_samples, save_dir):
    """
    Quantile-Quantile Plot
    https://stackoverflow.com/questions/46935289/quantile-quantile-plot-using-seaborn-and-scipy
    """
    # sns.set_theme(style="whitegrid")
    sns.despine()
    b = batch['precip_gt'].squeeze(0).cpu().numpy()
    # calculate quantiles
    percs = np.linspace(0, 100, 500)  # alter num to change density of plots
    qn_b = np.percentile(b, percs)
    # plot y = x ideal line
    x = np.linspace(np.min((qn_b.min(), qn_b.min())), np.max((qn_b.max(), qn_b.max())))
    sns.lineplot(x=x, y=x, color="k", ls="--")
    # plot dots
    for i in range(num_samples):
        a = batch['precip_output_' + str(i)].squeeze(0).cpu().numpy().flatten()
        qn_a = np.percentile(a, percs)
        sns.scatterplot(x=qn_a, y=qn_b, linewidth=0, label='Sample ' + str(i), alpha=0.7)
    plt.title('Quantile-Quantile Plot')
    plt.xlabel('Output (mm/h)')
    plt.ylabel('Ground Truth (mm/h)')
    # plt.legend()
    if save_dir is not None:
        plt.savefig(p.join(save_dir, f'qq_plot_ensemble.pdf'), dpi=600)
    else:
        plt.show()
    plt.close()
    return


def plot_qq_ensemble_2(batch, num_samples, save_dir):
    """
    Quantile-Quantile Plot
    https://stackoverflow.com/questions/46935289/quantile-quantile-plot-using-seaborn-and-scipy
    Plot samples together
    """
    # sns.set_theme(style="whitegrid")
    b = batch['precip_gt'].squeeze(0).cpu().numpy()
    axis_max = batch['precip_gt'].max().item()
    # calculate quantiles
    percs = np.linspace(0, 100, 500)  # alter num to change density of plots
    qn_b = np.percentile(b, percs)

    df = pd.DataFrame(columns=['Sample', 'Quantiles'])
    # Loop over each sample
    for i in range(num_samples):
        a = batch['precip_output_' + str(i)].squeeze(0).cpu().numpy().flatten()
        axis_max = max(axis_max, batch['precip_output_' + str(i)].max().item())
        qn_a = np.percentile(a, percs)

        # Create a DataFrame for the current sample
        df_sample = pd.DataFrame({
            'Sample': 'Sample ' + str(i),
            'sample_quantile': qn_a,
            'gt_quantile': qn_b
        })

        # Append the current sample's DataFrame to the main DataFrame
        df = pd.concat([df, df_sample], ignore_index=True)

    # Plot a single scatterplot using the main DataFrame
    # sns.scatterplot(data=df, x='sample_quantile', y='gt_quantile', linewidth=0, alpha=0.7)
    sns.regplot(data=df, x='sample_quantile', y='gt_quantile', order=3,
                scatter_kws={'alpha': 0.1}, marker=".")
    # plot y = x ideal line
    x = np.linspace(0, int(axis_max), 100)
    sns.lineplot(x=x, y=x, color="k", ls="--")
    # plot dots

    plt.xlim([0, axis_max])
    plt.ylim([0, axis_max])
    sns.despine()
    plt.title('Quantile-Quantile Plot')
    plt.xlabel('Output (mm/day)')
    plt.ylabel('Ground Truth (mm/day)')
    # plt.legend()
    if save_dir is not None:
        plt.savefig(p.join(save_dir, f'qq_plot_ensemble.pdf'), dpi=600)
        plt.show()
    else:
        plt.show()
    plt.close()
    return

def plot_psd(output, gt, save_dir, suffix=None):
    # power spectra distribution, individual output
    psd_output, freq_output = rapsd(output, return_freq=True, fft_method=np.fft, normalize=False)
    psd_gt, freq_gt = rapsd(gt, return_freq=True, fft_method=np.fft, normalize=False)

    # # Convert the PSDs to pandas DataFrames
    # psd_output_df = pd.DataFrame({'Frequency': freq_output, 'PSD': psd_output, 'Type': 'Output'})
    # psd_gt_df = pd.DataFrame({'Frequency': freq_gt, 'PSD': psd_gt, 'Type': 'Ground Truth'})
    #
    # # Concatenate the dataframes
    # psd_df = pd.concat([psd_output_df, psd_gt_df])


    fig, ax = plt.subplots()
    ax.plot(freq_output, psd_output, label='Output')
    ax.plot(freq_gt, psd_gt, label='Ground Truth')
    ax.set_xlabel('Frequency (1/km)')
    ax.set_ylabel('Rainfall intensity spectra (mm/day)')
    ax.set_xscale('log')  # Set log scale for x-axis
    ax.set_yscale('log')  # Set log scale for y-axis
    ax.set_title('Power Spectra')
    ax.legend()
    plt.grid(True)
    if save_dir is not None:
        suffix = '' if suffix is None else '_' + suffix
        plt.savefig(p.join(save_dir, f'psd{suffix}.png'))
    else:
        plt.show()
    plt.close()
    return


def plot_psd_ensemble(batch, num_samples, save_dir):
    # power spectra distribution, ensemble
    psd_gt, freq_gt = rapsd(batch['precip_gt'].squeeze(0).cpu().numpy(), return_freq=True,
                            fft_method=np.fft, normalize=False)
    fig, ax = plt.subplots()
    ax.plot(freq_gt, psd_gt, label='Ground Truth')
    for i in range(num_samples):
        output = batch['precip_output_' + str(i)].squeeze(0).cpu().numpy()
        psd_output, freq_output = rapsd(output, return_freq=True, fft_method=np.fft, normalize=False)
        ax.plot(freq_output, psd_output, label='Ouput ' + str(i))
    # Use seaborn to create the line plot
    ax.set_xlabel('Frequency (1/km)')
    ax.set_ylabel('Rainfall intensity spectra (mm/day)')
    ax.set_xscale('log')  # Set log scale for x-axis
    ax.set_yscale('log')  # Set log scale for y-axis
    ax.set_title('Power Spectra')
    ax.legend()
    plt.grid(True)
    if save_dir is not None:
        plt.savefig(p.join(save_dir, f'psd_ensemble.png'))
    else:
        plt.show()
    plt.close()
    return