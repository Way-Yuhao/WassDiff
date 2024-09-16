import os
import os.path as p
import sys
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
from hydra import compose, initialize
from natsort import natsorted
from src.data.cpc_mrms_dataset import DailyAggregateRainfallDataset
from scipy.stats.mstats import winsorize
import ast


"""
Plotting functions to run AFTER inference has been completed for a particular set of data
"""

# TODO: this file has not been adopted to LiT codebase
# TODO: there is a hydra error


def gather_quantile_data(dir_path, dataset, method_name, batch_key='precip_output'):
    batch = torch.load(p.join(dir_path, 'batch.pt'))
    batch = dataset.inverse_normalize_batch(batch)

    b = batch['precip_gt'].squeeze(0).cpu().numpy()
    axis_max = batch['precip_gt'].max().item()
    # calculate quantiles
    percs = np.linspace(0, 100, 500)  # alter num to change density of plots
    qn_b = np.percentile(b, percs)

    # df = pd.DataFrame(columns=['Sample', 'Quantiles'])
    df = pd.DataFrame(columns=['method', 'Sample', 'sample_quantile', 'gt_quantile'])
    # number of samples is the number of elements in batch with key 'precip_output'
    samples = [k for k in batch.keys() if batch_key in k]
    num_samples = len(samples)

    print('Found {} samples for {}'.format(num_samples, method_name))
    # Loop over each sample
    for i in range(num_samples):
        a = batch[samples[i]].squeeze(0).cpu().numpy().flatten()
        # axis_max = max(axis_max, batch['precip_output_' + str(i)].max().item())
        qn_a = np.percentile(a, percs)

        # Create a DataFrame for the current sample
        df_sample = pd.DataFrame({
            'method': method_name,
            'Sample': 'Sample ' + str(i),
            'sample_quantile': qn_a,
            'gt_quantile': qn_b
        })

        # Append the current sample's DataFrame to the main DataFrame
        df = pd.concat([df, df_sample], ignore_index=True)
    return df, axis_max


def plot_qq_ensemble(num_samples, save_dir):
    """
    Quantile-Quantile Plot
    https://stackoverflow.com/questions/46935289/quantile-quantile-plot-using-seaborn-and-scipy
    Plot samples together
    """
    sns.set_context('paper', font_scale=1.5)
    with initialize(version_base=None, config_path="../configs", job_name="evaluation"):
        config = compose(config_name="downscale_cpc_density")
        config.data.condition_mode = 6  # alter if needed
        config.data.image_size = 512
        config.data.condition_size = 512
        config.data.use_precomputed_era5 = False
        config.data.use_precomputed_cpc = False
    dataset = DailyAggregateRainfallDataset(config)
    parent_dir = '/home/yl241/workspace/NCSN/plt/'
    if not p.exists(save_dir):
        os.makedirs(save_dir)

    # WassDiff vs. SBDM
    # ours_dir = p.join(parent_dir, 'emdw_0.2_bill_vis_16')
    # our_minus_dir = p.join(parent_dir, 'no_emd_ckpt22_bill_16')
    #
    # ours_dir = p.join(parent_dir, 'emdw_0.2_cold_front_16')
    # our_minus_dir = p.join(parent_dir, 'no_emd_ckpt22_cold_front_16')

    # ours_dir = p.join(parent_dir, 'emd_wop2_gaint_hail_il_16')
    # our_minus_dir = p.join(parent_dir, 'no_emd_ckpt22_gaint_hail_il_16')

    # WassDiff vs. SBDM_r
    # ours_dir = p.join(parent_dir, 'emdw_0.2_bill_vis_16')
    # our_minus_dir = p.join(parent_dir, 'sbdm_r_bill_16')

    # ours_dir = p.join(parent_dir, 'emdw_0.2_cold_front_16')
    # our_minus_dir = p.join(parent_dir, 'sbdm_r_cold_front_16')

    ours_dir = p.join(parent_dir, 'emd_wop2_gaint_hail_il_16')
    our_minus_dir = p.join(parent_dir, 'sbdm_r_gaint_hail_il_16')

    ours_df, axis_max = gather_quantile_data(ours_dir, dataset, method_name='ours')
    df_ours_minus, _ = gather_quantile_data(our_minus_dir, dataset, method_name='ours-')
    cpc_inter, _ = gather_quantile_data(ours_dir, dataset, method_name='CPC_Int', batch_key='precip_up')

    # df = pd.concat([ours_df, ours_minus_dir], ignore_index=True)

    # Plot a single scatterplot using the main DataFrame
    # sns.scatterplot(data=df, x='sample_quantile', y='gt_quantile', linewidth=0, alpha=0.7)
    # sns.lmplot(data=df, x='sample_quantile', y='gt_quantile',
    #            order=3, hue='method', markers='.',
    #            scatter_kws={'alpha': 0.1})

    # sns.regplot(data=df_ours_minus, y='sample_quantile', x='gt_quantile', order=3,
    #             scatter_kws={'alpha': 0.1}, marker=".",x_estimator=np.mean)

    sns.lineplot(data=ours_df, y='sample_quantile', x='gt_quantile', color='tab:blue',
                 markers='o', errorbar='sd', linewidth=2)
    sns.lineplot(data=df_ours_minus, y='sample_quantile', x='gt_quantile', color='tab:purple',
                 markers='o', errorbar='sd', linewidth=2)
    # sns.lineplot(data=cpc_inter, y='sample_quantile', x='gt_quantile', color='tab:green')



    # plot y = x ideal line
    x = np.linspace(0, int(axis_max), 100)
    sns.lineplot(x=x, y=x, color="k", ls="--", linewidth=2)
    # plot dots

    # plt.xlim([0, axis_max])
    # plt.ylim([0, axis_max])
    sns.despine()
    plt.title('Quantile-Quantile Plot')
    plt.ylabel('Output (mm/day)')
    plt.xlabel('Ground Truth (mm/day)')
    # plt.legend()
    if save_dir is not None:
        plt.savefig(p.join(save_dir, f'hail_r.pdf'), dpi=600)
        plt.show()
    else:
        plt.show()
    plt.close()
    return


def dist_output_specific_sample():
    sns.set_context('paper')
    # make the figure thinner
    plt.figure(figsize=(2, 5))
    ours_path = '/home/yl241/workspace/NCSN/plt/emdw_0.2_bill_vis_16'
    ours_minus_path = '/home/yl241/workspace/NCSN/plt/no_emd_ckpt22_bill_16'
    batch = torch.load(p.join(ours_path, 'batch.pt'))
    df = pd.DataFrame(columns=['method', 'mean'])
    means = []
    for i in range(16):
        means += [batch[f'precip_output_{i}'].mean().item()]
    df = pd.DataFrame({'method': 'ours', 'mean': means})

    batch = torch.load(p.join(ours_minus_path, 'batch.pt'))
    means = []
    for i in range(16):
        means += [batch[f'precip_output_{i}'].mean().item()]
    df = pd.concat([df, pd.DataFrame({'method': 'ours_minus', 'mean': means})], ignore_index=True)
    print(df)
    df['mean'] = winsorize(df['mean'], limits=[0.05, 0.05])
    sns.kdeplot(df, y='mean', hue='method')
    plt.show()


def dist_mean_prior():
    plt.figure(figsize=(2, 5))
    means = []
    for i in range(26 * 12):
        mu = (torch.randn((1, 1, 256, 256)) * 50).mean().item()
        means += [mu]
    df = pd.DataFrame({'method': 'prior', 'mean': means})
    palette = {'prior': 'sienna'}
    sns.kdeplot(data=df, y='mean', hue='method', palette=palette,
                bw_adjust=.99)
    plt.ylim([-0.8, 1.0])
    plt.legend().remove()
    plt.savefig('/home/yl241/data/rainfall_eval/general/prior_mean.pdf', dpi=600)
    plt.show()
    plt.close()


def dist_mean_val_set():
    ours_path = '/home/yl241/data/rainfall_eval/logp1_emd_ckpt21'
    ours_minus_path = '/home/yl241/data/rainfall_eval/logp1_ckpt22'
    plt.figure(figsize=(2, 5))
    with initialize(version_base=None, config_path="../configs", job_name="evaluation"):
        config = compose(config_name="downscale_cpc_density")
        config.data.condition_mode = 6  # alter if needed
        config.data.image_size = 512
        config.data.condition_size = 512
        config.data.use_precomputed_era5 = False
        config.data.use_precomputed_cpc = False
    dataset = DailyAggregateRainfallDataset(config)

    # df = pd.DataFrame(columns=['method', 'mean'])
    means = []
    for i in range(26):
        batch = torch.load(p.join(ours_path, f'batch_{i}.pt'))
        mean = batch['precip_output'].mean().item()
        mean = np.log(mean + 0.00001) / 5
        means += [mean]
    df = pd.DataFrame({'method': 'ours', 'mean': means})

    means = []
    for i in range(26):
        batch = torch.load(p.join(ours_path, f'batch_{i}.pt'))
        mean = batch['precip_gt'].mean().item()
        mean = np.log(mean + 0.00001) / 5
        means += [mean]
    df = pd.concat([df, pd.DataFrame({'method': 'gt', 'mean': means})], ignore_index=True)

    means = []
    for i in range(26):
        batch = torch.load(p.join(ours_minus_path, f'batch_{i}.pt'))
        mean = batch['precip_output'].mean().item()
        mean = np.log(mean + 0.00001) / 5
        means += [mean]

    df = pd.concat([df, pd.DataFrame({'method': 'ours_minus', 'mean': means})], ignore_index=True)
    palette = {'ours': 'tab:blue', 'ours_minus': 'tab:purple', 'gt': 'black'}
    sns.kdeplot(data=df, y='mean', hue='method', palette=palette)
    plt.ylim([-0.7, 1.1])
    plt.legend().remove()
    plt.savefig('/home/yl241/data/rainfall_eval/general/mean_val_set.pdf', dpi=600)
    # plt.show()


def load_data_from_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                # Convert string representation of list to a Python list
                parsed_list = ast.literal_eval(line)
                data.append(parsed_list)

    # Convert list of lists to a NumPy array
    numpy_array = np.array(data)

    # Remove singleton dimensions
    numpy_array = np.squeeze(numpy_array)
    return numpy_array


def show_sde_trajecotry():
    sns.set_context('paper')
    # read data from '/home/yl241/data/rainfall_eval/ours_trajectory.txt'
    data_score_diff = load_data_from_file('/home/yl241/data/rainfall_eval/general/sde_trajectory_1.txt')
    data_wass_diff = load_data_from_file('/home/yl241/data/rainfall_eval/general/sde_trajectory.txt')
    data_score_diff = np.array(data_score_diff)
    data_wass_diff = np.array(data_wass_diff)
    # plt.plot(data_score_diff[:, 4], c='tab:purple', linestyle='--')
    # plt.plot(data_wass_diff, c='tab:blue', linestyle='--')


    plt.plot(data_score_diff[:, 0], c='tab:purple', linestyle='--')
    plt.plot(data_score_diff[:, 1], c='tab:purple', linestyle='--')
    plt.plot(data_score_diff[:, 2], c='tab:purple', linestyle='--')
    # plt.plot(data_score_diff[:, 3], label='4', c='tab:blue')
    plt.plot(data_score_diff[:, 4], c='tab:purple', linestyle='--')

    plt.plot(data_wass_diff[:, 0], c='tab:blue', linestyle='--')
    plt.plot(data_wass_diff[:, 1], c='tab:blue', linestyle='--')
    plt.plot(data_wass_diff[:, 2], c='tab:blue', linestyle='--')
    # plt.plot(data_wass_diff[:, 3],
    #          c='tab:blue')
    plt.plot(data_wass_diff[:, 4], c='tab:blue', linestyle='--')


    plt.ylim([-0.7, 1.1])
    plt.xlim([0, 200])


    plt.savefig('/home/yl241/data/rainfall_eval/general/sde_trajectory.pdf', dpi=600)
    plt.show()

    plt.close()
    return


def main():
    # plot_qq_ensemble(16, '/home/yl241/workspace/NCSN/plt/qq')
    # dist_output_specific_sample()
    # dist_mean_prior()
    # dist_mean_val_set()
    # show_sde_trajecotry()
    pass


if __name__ == '__main__':
    main()
