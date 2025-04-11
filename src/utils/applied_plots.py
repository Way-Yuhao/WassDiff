import os
import os.path as p
# import sys
import numpy as np
# import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
from hydra import compose, initialize
from natsort import natsorted
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tqdm import tqdm
from src.data.cpc_mrms_dataset import DailyAggregateRainfallDataset
from scipy.stats.mstats import winsorize
import ast
from src.utils.callbacks.eval_on_dataset import compute_ensemble_metrics
from src.utils.plot_func import filter_sample_logic, build_histogram_for_sample, build_power_spectral_for_sample
from src.utils.helper import yprint

"""
META ANALYSIS PLOTS
Plotting functions to run AFTER inference has been completed for a particular set of data
"""
# TODO: this file has not been adopted to LiT codebase
# TODO: there is a hydra error


def gather_quantile_data(dir_path, dataset, method_name, batch_key='precip_output'):
    batch = torch.load(p.join(dir_path, 'batch.pt'))
    if type(batch) is tuple:
        batch = batch[0]
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


def calc_mppe_ensemble_mean(dir_path, dataset, batch_key='precip_output'):
    # Load the batch and inverse-normalize it
    batch = torch.load(p.join(dir_path, 'batch.pt'))
    if isinstance(batch, tuple):
        batch = batch[0]
    batch = dataset.inverse_normalize_batch(batch)
    # Extract ground truth
    gt = batch['precip_gt'].squeeze(0).cpu().numpy()
    # Gather ensemble outputs based on the batch_key
    sample_keys = [k for k in batch.keys() if batch_key in k]
    outputs = []
    for key in sample_keys:
        output = batch[key].squeeze(0).cpu().numpy()
        outputs.append(output)
    # Compute the ensemble mean field pixel-wise
    ensemble_mean = np.mean(np.stack(outputs, axis=0), axis=0)
    # Compute the 99.9th percentiles
    ensemble_999 = np.percentile(ensemble_mean, 99.9)
    gt_999 = np.percentile(gt, 99.9)
    # Compute MPPE; since it's a scalar difference, this is equivalent to the absolute error
    mppe = np.sqrt(np.mean((ensemble_999 - gt_999) ** 2))
    return mppe


def plot_qq_ensemble(num_samples, save_dir):
    """
    Quantile-Quantile Plot
    https://stackoverflow.com/questions/46935289/quantile-quantile-plot-using-seaborn-and-scipy
    Plot samples together
    """
    sns.set_context('paper', font_scale=1.5)
    with initialize(version_base=None, config_path="../../configs/", job_name="evaluation"):
        config = compose(config_name="train")
        config.data.data_config.condition_mode = 6  # alter if needed
        config.data.data_config.image_size = 512
        config.data.data_config.condition_size = 512
        config.data.data_config.use_precomputed_era5 = False
        config.data.data_config.use_precomputed_cpc = False
    dataset = DailyAggregateRainfallDataset(config.data.data_config)
    parent_dir = '/home/yl241/data/rainfall_plots_LiT/'
    if not p.exists(save_dir):
        os.makedirs(save_dir)

    ## WassDiff vs. SBDM
    # ours_dir = p.join(parent_dir, 'emdw_0.2_bill_vis_16')
    # our_minus_dir = p.join(parent_dir, 'no_emd_ckpt22_bill_16')
    #
    # ours_dir = p.join(parent_dir, 'emdw_0.2_cold_front_16')
    # our_minus_dir = p.join(parent_dir, 'no_emd_ckpt22_cold_front_16')

    # ours_dir = p.join(parent_dir, 'emd_wop2_gaint_hail_il_16')
    # our_minus_dir = p.join(parent_dir, 'no_emd_ckpt22_gaint_hail_il_16')

    ## WassDiff vs. SBDM_r
    # ours_dir = p.join(parent_dir, 'emdw_0.2_bill_vis_16')
    # our_minus_dir = p.join(parent_dir, 'sbdm_r_bill_16')

    # ours_dir = p.join(parent_dir, 'emdw_0.2_cold_front_16')
    # our_minus_dir = p.join(parent_dir, 'sbdm_r_cold_front_16')

    # ours_dir = p.join(parent_dir, 'emd_wop2_gaint_hail_il_16')
    # our_minus_dir = p.join(parent_dir, 'sbdm_r_gaint_hail_il_16')

    ## WassDiff vs. SBDM_r vs. CorrDiff
    # ours_dir = p.join(parent_dir, 'emdw_0.2_bill_vis_16')
    # our_minus_dir = p.join(parent_dir, 'sbdm_r_bill_16')
    # corrdiff_dir = p.join(parent_dir, 'corrdiff_bill_16')

    # ours_dir = p.join(parent_dir, 'emdw_0.2_cold_front_16')
    # our_minus_dir = p.join(parent_dir, 'sbdm_r_cold_front_16')
    # corrdiff_dir = p.join(parent_dir, 'corrdiff_cold_front_16')

    ours_dir = p.join(parent_dir, 'emd_wop2_gaint_hail_il_16')
    our_minus_dir = p.join(parent_dir, 'sbdm_r_gaint_hail_il_16')
    corrdiff_dir = p.join(parent_dir, 'corrdiff_giant_hill_16')


    ours_df, axis_max = gather_quantile_data(ours_dir, dataset, method_name='ours')
    df_ours_minus, _ = gather_quantile_data(our_minus_dir, dataset, method_name='ours-')
    df_corrdiff, _ = gather_quantile_data(corrdiff_dir, dataset, method_name='CorrDiff')
    cpc_inter, _ = gather_quantile_data(ours_dir, dataset, method_name='CPC_Int', batch_key='precip_up')

    mppe_ours = calc_mppe_ensemble_mean(ours_dir, dataset)
    mppe_ours_minus = calc_mppe_ensemble_mean(our_minus_dir, dataset)
    mppe_corrdiff = calc_mppe_ensemble_mean(corrdiff_dir, dataset)
    print(f'MPPE for ours: {mppe_ours}')
    print(f'MPPE for ours-: {mppe_ours_minus}')
    print(f'MPPE for CorrDiff: {mppe_corrdiff}')

    sns.lineplot(data=ours_df, y='sample_quantile', x='gt_quantile', color='tab:blue',
                 markers='o', errorbar='sd', linewidth=0.5)
    sns.lineplot(data=df_ours_minus, y='sample_quantile', x='gt_quantile', color='tab:purple',
                 markers='o', errorbar='sd', linewidth=0.5)
    sns.lineplot(data=df_corrdiff, y='sample_quantile', x='gt_quantile', color='saddlebrown',
                    markers='o', errorbar='sd', linewidth=0.5)
    # sns.lineplot(data=cpc_inter, y='sample_quantile', x='gt_quantile', color='tab:green')

    ax = plt.gca()
    ax.tick_params(width=0.5, length=3)

    # plot y = x ideal line
    x = np.linspace(0, int(axis_max), 100)
    sns.lineplot(x=x, y=x, color="k", ls="--", linewidth=0.5)
    # plot dots

    # plt.xlim([0, axis_max])
    # plt.ylim([0, axis_max])
    sns.despine()
    # plt.title('Quantile-Quantile Plot')
    plt.ylabel('Output (mm/day)')
    plt.xlabel('Ground Truth (mm/day)')
    # plt.legend()
    if save_dir is not None:
        plt.savefig(p.join(save_dir, f'bill_corrdiff.svg'), dpi=300, transparent=True)
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


def show_sde_trajectory():
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


def skill_vs_ensemble_size():

    def _plot_line(df: pd.DataFrame, metric: str, *args, **kwargs):
        df_for_metric = df[df['Metrics'] == metric.lower()]
        sns.pointplot(data=df_for_metric, x='ensemble_size', y='Mean', errorbar=None, label=metric, *args, **kwargs)
        return

    eval_root_dir = '/home/yl241/data/rainfall_eval'
    output_dir_signature = 'logp1_emd_ckpt21'  # folder name must partially match this signature
    ensemble_dir = '/home/yl241/data/rainfall_eval/logp1_emd_ckpt21_ensemble'

    # scan for output dirs
    output_dirs = os.listdir(eval_root_dir)
    output_dirs = [d for d in output_dirs if output_dir_signature in d]
    output_dirs = [d for d in output_dirs if '+' not in d]
    output_dirs = natsorted([d for d in output_dirs if 'ensemble' not in d])
    print('Max ensemble size = {} for outputs matching signature {}'.format(len(output_dirs), output_dir_signature))
    max_ensemble_size = len(output_dirs)
    # print(output_dirs)

    ## collect metrics for different ensemble sizes
    # for m in range(1, max_ensemble_size + 1):
    #     compute_ensemble_metrics(parent_save_dir=eval_root_dir,
    #                              save_dir_pattern=output_dir_signature,
    #                              show_vis=False, show_metric_for='ours',
    #                              ensemble_size=m)


    ## consolidate dataframes for mean and std
    # individual_dfs = 'summary_stats_{}_members.csv'
    # consolidated_df = pd.DataFrame()
    # for m in trange(1, max_ensemble_size + 1, desc='Consolidating dataframes...'):
    #     df = pd.read_csv(p.join(ensemble_dir, individual_dfs.format(m)))
    #     # add ensemble size column
    #     df['ensemble_size'] = m
    #     consolidated_df = pd.concat([consolidated_df, df], ignore_index=True)
    # consolidated_df.rename(columns={'Unnamed: 0': 'Metrics'}, inplace=True)
    # consolidated_df.to_csv(p.join(ensemble_dir, 'summary_stats_all_members.csv'), index=False)


    ## consodilate dataframes for individual datapoints
    # individual_dfs = 'metrics_{}_members.csv'
    # consolidated_df = pd.DataFrame()
    # for m in trange(1, max_ensemble_size + 1, desc='Consolidating dataframes...'):
    #     df = pd.read_csv(p.join(ensemble_dir, individual_dfs.format(m)))
    #     # add ensemble size column
    #     df['ensemble_size'] = m
    #     consolidated_df = pd.concat([consolidated_df, df], ignore_index=True)
    # consolidated_df.rename(columns={'Unnamed: 0': 'Metrics'}, inplace=True)
    # consolidated_df.to_csv(p.join(ensemble_dir, 'metrics_all_members.csv'), index=False)


    ## plot in terms of mean and std
    # consolidated_df = pd.read_csv(p.join(ensemble_dir, 'summary_stats_all_members.csv'))
    # _plot_line(consolidated_df, 'MAE', color='slategrey')
    # _plot_line(consolidated_df, 'CRPS', color='brown')
    # plt.legend()
    # sns.despine()
    # plt.show()

    ## plot in terms of individual datapoints
    consolidated_df = pd.read_csv(p.join(ensemble_dir, 'metrics_all_members.csv'))
    metric = 'CRPS'
    # sns.catplot(data=consolidated_df, x='ensemble_size', y=metric.lower(), color='slategrey',
    #             errorbar="se", kind='point')

    # sns.catplot(data=consolidated_df, x='ensemble_size', y=metric.lower(), color='slategrey',
    #             errorbar="se", kind='point')

    # sns.violinplot(data=consolidated_df, x='ensemble_size', y=metric.lower(), split=True, inner="quart",
    #                cut=0)

    sns.boxenplot(data=consolidated_df, x='ensemble_size', y=metric.lower(), showfliers=False)

    sns.despine()
    plt.show()


def build_hist_for_all_methods(ensemble_size: int, graph_to_build: str):
    """
    Build histograms for all methods
    :param ensemble_size: number of ensemble members
    :param graph_to_build: 'hist' or 'spectra'
    """
    print('Building histograms for all methods with ensemble size ', ensemble_size)
    sns.set_context('paper', font_scale=1.5)

    out_dir = '/home/yl241/data/rainfall_eval_LiT/general'
    if not p.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Created directory: {out_dir}')
    all_methods = {
        'Ours': '/home/yl241/data/rainfall_eval/logp1_emd_ckpt21',
        # 'Ours-': '/home/yl241/data/rainfall_eval/logp1_ckpt22',
        'Ours-': '/home/yl241/data/rainfall_eval/sbdm_r',
        'CNN': '/home/yl241/data/rainfall_eval/cnn_baseline_r21ckpt',
        'CGAN': '/home/yl241/data/rainfall_eval_LiT/CorrectorGAN_epoch_699',
        'CorrDiff': '/home/yl241/data/rainfall_eval_LiT_rebuttal/CorrDiff_ep399'
    }
    label_colors = {
        'Ours': 'tab:blue',
        'Ours-': 'tab:purple',
        'CNN': 'tab:green',
        'CPC_Int': 'tab:orange',
        'Ground Truth': 'black',
        'CGAN': 'tab:red',
        'CorrDiff': 'saddlebrown'
    }

    ig, ax = plt.subplots(figsize=(8, 4.5))
    for method, method_dir in all_methods.items():
        hist_total = None
        spectra_total = None
        if method == 'Ours-' or method == 'Ours':
            files = []
            for i in range(1, ensemble_size+1):
                if i > 1:
                    method_dir_ = method_dir + f'_{i}'
                else:
                    method_dir_ = method_dir
                additional_files = os.listdir(method_dir_)
                additional_files = [p.join(method_dir_, f) for f in additional_files if f.endswith('.pt')]
                files += additional_files

        else:
            files = os.listdir(method_dir)
            files = [p.join(method_dir, f) for f in files if f.endswith('.pt')]
        batch_size = torch.load(files[0])['precip_up'].shape[0]
        for f in tqdm(files, desc=f'Building histograms for {method}'):
            batch_dict = torch.load(f)
            output_batch = batch_dict['precip_output'].cpu().detach().numpy()
            gt_batch = batch_dict['precip_gt'].cpu().detach().numpy()
            for i in range(batch_size):
                output = output_batch[i][0, :, :]
                gt = gt_batch[i][0, :, :]
                valid_mask = gt != -1
                if not filter_sample_logic(valid_mask, logic='remove_any_invalid'):
                    continue
                hist_output, bin_edges_output = build_histogram_for_sample(output[valid_mask])
                spectra_output, freq = build_power_spectral_for_sample(output)
                if hist_total is None:
                    hist_total = hist_output
                    spectra_total = spectra_output
                else:
                    hist_total += hist_output
                    spectra_total += spectra_output

        if graph_to_build == 'hist': # build histograms
            # normalize histograms to convert to pdf
            hist_total = hist_total / hist_total.sum()
            ax.plot(bin_edges_output[:-1], hist_total, label=method, color=label_colors[method], linewidth=2)
        elif graph_to_build == 'spectra':  # build power spectral density plot
            if method in ['Ours', 'Ours-']:
                ax.plot(freq, spectra_total / ensemble_size, label=method, color=label_colors[method], linewidth=2)
            else:
                ax.plot(freq, spectra_total, label=method, color=label_colors[method], linewidth=2)
    # handle ours and cpc_int
    gt_hist_total, cpc_inter_hist_total, output_hist_total = None, None, None
    gt_spectra_total, cpc_spectra_total, output_spectral_total = None, None, None
    files = os.listdir(all_methods['Ours'])
    files = [f for f in files if f.endswith('.pt')]
    batch_size = torch.load(os.path.join(all_methods['Ours'], files[0]))['precip_up'].shape[0]
    for f in tqdm(files, desc=f'Building histograms for CPC and MRMS'):
        batch_dict = torch.load(p.join(method_dir, f))
        cpc_inter_batch = batch_dict['precip_up'].cpu().detach().numpy()
        gt_batch = batch_dict['precip_gt'].cpu().detach().numpy()
        for i in range(batch_size):
            cpc_inter = cpc_inter_batch[i][0, :, :]
            gt = gt_batch[i][0, :, :]
            valid_mask = gt != -1
            if not filter_sample_logic(valid_mask, logic='remove_any_invalid'):
                continue
            hist_gt, bin_edges_gt = build_histogram_for_sample(gt[valid_mask])
            hist_cpc, bin_edges_cpc = build_histogram_for_sample(cpc_inter[valid_mask])
            spectra_cpc, freq = build_power_spectral_for_sample(cpc_inter)
            spectra_gt, _ = build_power_spectral_for_sample(gt)
            if gt_hist_total is None:
                gt_hist_total = hist_gt
                cpc_inter_hist_total = hist_cpc
                gt_spectra_total = spectra_gt
                cpc_spectra_total = spectra_cpc
            else:
                gt_hist_total += hist_gt
                cpc_inter_hist_total += hist_cpc
                gt_spectra_total += spectra_gt
                cpc_spectra_total += spectra_cpc

    if graph_to_build == 'hist': # normalize histograms to convert to pdf
        gt_hist_total = gt_hist_total / gt_hist_total.sum()
        cpc_inter_hist_total = cpc_inter_hist_total / cpc_inter_hist_total.sum()
        # build histograms
        ax.plot(bin_edges_gt[:-1], gt_hist_total, label='Ground Truth', color=label_colors['Ground Truth'], linewidth=2)
        ax.plot(bin_edges_cpc[:-1], cpc_inter_hist_total, label='CPC_Int', color=label_colors['CPC_Int'], linewidth=2)
    elif graph_to_build == 'spectra':
        # build power spectral density plot
        ax.plot(freq, gt_spectra_total, label='Ground Truth', color=label_colors['Ground Truth'], linewidth=2)
        ax.plot(freq, cpc_spectra_total, label='CPC_Int', color=label_colors['CPC_Int'], linewidth=2)

    if graph_to_build == 'hist': # histogram
        ax.set_xlabel('Rainfall intensity (mm/day)')
        ax.set_ylabel('PDF')
        ax.set_yscale('log')  # Set log scale for y-axis
        # ax.set_title('Histograms')
        # ax.legend()
        sns.despine()
        plt.tight_layout()
        plt.savefig(p.join(out_dir, 'histograms_r.pdf'))
        plt.show()
        plt.close()
    elif graph_to_build == 'spectra': # power spectral density
        ax.set_xlabel('Frequency (1/km)')
        ax.set_ylabel('Power Spectra')
        ax.set_xscale('log')  # Set log scale for x-axis
        ax.set_yscale('log')  # Set log scale for y-axis
        # ax.set_title('Power Spectra')
        # ax.legend()
        plt.tight_layout()
        sns.despine()
        plt.savefig(p.join(out_dir, 'spectra_r.pdf'))
        plt.show()
        plt.close()

def plot_additional_vis():
    output_dir = '/home/yl241/data/rainfall_eval_LiT/general/vis_with_corrdiff/'
    if not p.exists(output_dir):
        os.makedirs(output_dir)

    # ours_dir = '/home/yl241/data/rainfall_eval/logp1_emd_ckpt21'
    # ours_minus_dir = '/home/yl241/data/rainfall_eval/sbdm_r'
    # cnn_dir = '/home/yl241/data/rainfall_eval/cnn_baseline_r21ckpt'
    # cgan_dir = '/home/yl241/data/rainfall_eval_LiT/CorrectorGAN_epoch_699'
    ours_dir = '/home/yl241/data/rainfall_eval_LiT_rebuttal/logp1_emd_ckpt21'
    ours_minus_dir = '/home/yl241/data/rainfall_eval_LiT_rebuttal/sbdm_r'
    cnn_dir = '/home/yl241/data/rainfall_eval_LiT_rebuttal/cnn_baseline_ckpt21r_rebuttal'
    cgan_dir = '/home/yl241/data/rainfall_eval_LiT_rebuttal/CorrectorGAN_epoch_699'
    corrdiff_dir = '/home/yl241/data/rainfall_eval_LiT_rebuttal/CorrDiff_ep399'
    num_batches = 40
    batch_size = 12
    for i in tqdm(range(num_batches), desc='Plotting visualizations'):
        for j in range(batch_size):
            ours_batch = torch.load(p.join(ours_dir, f'batch_{i}.pt'))
            ours_minus_batch = torch.load(p.join(ours_minus_dir, f'batch_{i}.pt'))
            cnn_batch = torch.load(p.join(cnn_dir, f'batch_{i}.pt'))
            cgan_batch = torch.load(p.join(cgan_dir, f'batch_{i}.pt'))
            corrdiff_batch = torch.load(p.join(corrdiff_dir, f'batch_{i}.pt'))


            ours = ours_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            ours_minus = ours_minus_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            cnn = cnn_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            cpc_inter = ours_batch['precip_up'][j][0, :, :].cpu().detach().numpy()
            cgan = cgan_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            corrdiff = corrdiff_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            gt = ours_batch['precip_gt'][j][0, :, :].cpu().detach().numpy()

            if gt.max() > 0:
                # 99 percentile of gt
                vmax_ = np.percentile(gt, 99.9)
                # vmax_ = gt.max()
                vmin_ = 0
            else:
                vmax_ = max(cpc_inter.max(), ours.max(), gt.max())
                vmin_ = min(cpc_inter.min(), ours.min(), gt.min())

            fig, axes = plt.subplots(1, 7, figsize=(22, 5))
            im1 = axes[0].imshow(cpc_inter, cmap='viridis', vmin=vmin_, vmax=vmax_)
            im2 = axes[1].imshow(cnn, cmap='viridis', vmin=vmin_, vmax=vmax_)
            im3 = axes[2].imshow(cgan, cmap='viridis', vmin=vmin_, vmax=vmax_)
            im4 = axes[3].imshow(corrdiff, cmap='viridis', vmin=vmin_, vmax=vmax_)
            im5 = axes[4].imshow(ours_minus, cmap='viridis', vmin=vmin_, vmax=vmax_)
            im6 = axes[5].imshow(ours, cmap='viridis', vmin=vmin_, vmax=vmax_)
            im7 = axes[6].imshow(gt, cmap='viridis', vmin=vmin_, vmax=vmax_)
            cbar = fig.colorbar(im5, ax=axes, shrink=0.45, pad=0.01)
            cbar.set_label("Precipitation\n(mm/day)", fontsize=12)
            for ax in axes:
                ax.axis('off')

            # SVG only: Reduce the colorbar outline thickness
            cbar.outline.set_linewidth(0.75)  # Set to a smaller value (default is ~1.5)
            cbar.ax.tick_params(width=0.75)

            plt.savefig(p.join(output_dir, f'batch_{i}_sample_{j}.svg'),
                        bbox_inches='tight', dpi=600, transparent=True)
            # plt.show()
            plt.close()

def plot_additional_vis_era5_ablation():
    output_dir = '/home/yl241/data/rainfall_eval_LiT/general/vis_ablation/'
    ours_dir = '/home/yl241/data/rainfall_eval/logp1_emd_ckpt21'
    data_root_dir = '/home/yl241/data/rainfall_eval_LiT'
    precip_only_dir = os.path.join(data_root_dir, 'WassDiff_ablation_precip_only')
    no_precip_dir = os.path.join(data_root_dir, 'WassDiff_ablation_no_precip_up')
    no_density_dir = os.path.join(data_root_dir, 'WassDiff_ablation_density')
    no_surf_temp_dir = os.path.join(data_root_dir, 'WassDiff_ablation_surf_temp')
    no_elevation_dir = os.path.join(data_root_dir, 'WassDiff_ablation_elevation')
    no_wind_dir = os.path.join(data_root_dir, 'WassDiff_ablation_no_wind_u_v')
    no_vflux_dir = os.path.join(data_root_dir, 'WassDiff_ablation_vlux_e_n')

    num_batches = 25
    batch_size = 12
    for i in tqdm(range(num_batches), desc='Plotting visualizations'):
        for j in range(batch_size):

            ours_batch = torch.load(p.join(ours_dir, f'batch_{i}.pt'))
            precip_only_batch = torch.load(p.join(precip_only_dir, f'batch_{i}.pt'))
            no_precip_batch = torch.load(p.join(no_precip_dir, f'batch_{i}.pt'))
            no_density_batch = torch.load(p.join(no_density_dir, f'batch_{i}.pt'))
            no_surf_temp_batch = torch.load(p.join(no_surf_temp_dir, f'batch_{i}.pt'))
            no_elevation_batch = torch.load(p.join(no_elevation_dir, f'batch_{i}.pt'))
            no_wind_batch = torch.load(p.join(no_wind_dir, f'batch_{i}.pt'))
            no_vflux_batch = torch.load(p.join(no_vflux_dir, f'batch_{i}.pt'))

            ours = ours_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            cpc_inter = ours_batch['precip_up'][j][0, :, :].cpu().detach().numpy()
            precip_only = precip_only_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            no_precip = no_precip_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            no_density = no_density_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            no_surf_temp = no_surf_temp_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            no_elevation = no_elevation_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            no_wind = no_wind_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            no_vflux = no_vflux_batch['precip_output'][j][0, :, :].cpu().detach().numpy()
            gt = ours_batch['precip_gt'][j][0, :, :].cpu().detach().numpy()

            if gt.max() > 0:
                # 99 percentile of gt
                vmax_ = np.percentile(gt, 99.9)
                # vmax_ = gt.max()
                vmin_ = 0
            else:
                vmax_ = max(cpc_inter.max(), ours.max(), gt.max())
                vmin_ = min(cpc_inter.min(), ours.min(), gt.min())
            # plot in the order of cpc_inter, cnn, ours-, ours
            fig, axes = plt.subplots(1, 10, figsize=(14, 3))
            # axes[0].imshow(cpc_inter, cmap='viridis', vmin=vmin_, vmax=vmax_)
            axes[1].imshow(precip_only, cmap='viridis', vmin=vmin_, vmax=vmax_)
            axes[2].imshow(no_precip, cmap='viridis', vmin=vmin_, vmax=vmax_)
            axes[3].imshow(no_density, cmap='viridis', vmin=vmin_, vmax=vmax_)
            axes[4].imshow(no_surf_temp, cmap='viridis', vmin=vmin_, vmax=vmax_)
            axes[5].imshow(no_elevation, cmap='viridis', vmin=vmin_, vmax=vmax_)
            axes[6].imshow(no_wind, cmap='viridis', vmin=vmin_, vmax=vmax_)
            axes[7].imshow(no_vflux, cmap='viridis', vmin=vmin_, vmax=vmax_)
            axes[8].imshow(ours, cmap='viridis', vmin=vmin_, vmax=vmax_)
            im9 = axes[9].imshow(gt, cmap='viridis', vmin=vmin_, vmax=vmax_)
            cbar = fig.colorbar(im9, ax=axes, shrink=0.31, pad=0.01)
            cbar.set_label("Precipitation\n(mm/day)", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
            for ax in axes:
                ax.axis('off')
            # add colorbar at the right of the last image, applies to all images
            plt.savefig(p.join(output_dir, f'batch_{i}_sample_{j}.pdf'),
                        bbox_inches='tight', dpi=600, transparent=True)
            # plt.show()
            plt.close()


def sample_bias_during_training():
    # Read and preprocess df1
    df1 = pd.read_csv('/home/yl241/data/rainfall_eval_LiT/general/sbdm_sample_bias.csv')
    df1 = df1.rename(columns={'Ablation WDR - val/sample_bias': 'sample_bias'})
    df1 = df1[['Step', 'sample_bias']]
    df1['Method'] = 'SBDM'  # Add method name column

    # Read and preprocess df2
    df2 = pd.read_csv('/home/yl241/data/rainfall_eval_LiT/general/wassdiff_sample_bias.csv')
    df2 = df2.rename(columns={'WassDiff det val - val/sample_bias': 'sample_bias'})
    df2 = df2[['Step', 'sample_bias']]
    df2['Method'] = 'WassDiff'  # Add method name column

    # Concatenate the DataFrames
    df = pd.concat([df1, df2], ignore_index=True)

    # Plot using Seaborn
    sns.lineplot(data=df, x='Step', y='sample_bias', hue='Method')

    # Optional: Add labels and title
    plt.xlabel('Step')
    plt.ylim([0, 200])
    plt.ylabel('Sample Bias')
    plt.title('Sample Bias over Steps for Different Methods')

    # Display the plot
    plt.show()

def main():
    # plot_qq_ensemble(16, '/home/yl241/data/rainfall_plots_LiT/general/qq/')
    # dist_output_specific_sample()
    # dist_mean_prior()
    # dist_mean_val_set()
    # show_sde_trajectory()
    # skill_vs_ensemble_size()

    # hist and spectra
    # build_hist_for_all_methods(ensemble_size=13, graph_to_build='hist')
    # build_hist_for_all_methods(ensemble_size=13, graph_to_build='spectra')

    plot_additional_vis()
    # plot_additional_vis_era5_ablation()
    # sample_bias_during_training()
if __name__ == '__main__':
    main()
