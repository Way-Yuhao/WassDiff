import os
import os.path as p
from typing import Any, Dict, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt
from natsort import natsorted
from datetime import datetime
from lightning.pytorch.callbacks import RichProgressBar, Callback
from lightning.pytorch.utilities import rank_zero_only
from src.utils.helper import yprint, monitor_complete
import lpips
from src.utils.metrics import calc_mae, calc_mse, calc_rmse, calc_pcc, calc_csi, calc_bias, calc_fss, calc_emd, \
    calc_hrre, calc_mppe, calc_lpips, calc_crps


class EvalOnDataset(Callback):
    """
    Callback for evaluating on a specified dataset.
    """

    def __init__(self, save_dir: str, skip_existing: bool, eval_only_on_existing: bool, show_vis: bool,
                 csi_threshold: float, heavy_rain_threshold: float, peak_mesoscale_threshold: float):
        super().__init__()
        self.save_dir = save_dir
        self.skip_existing = skip_existing
        self.eval_only_on_existing = eval_only_on_existing
        self.show_vis = show_vis
        if self.show_vis:
            self.vis_dir = p.join(save_dir, 'vis')
            os.makedirs(self.vis_dir, exist_ok=True)

        # metrics-related constants
        self.csi_threshold = csi_threshold
        self.heavy_rain_threshold = heavy_rain_threshold
        self.peak_mesoscale_threshold = peak_mesoscale_threshold

        # to be defined elsewhere
        self.rainfall_dataset = None
        return

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.rainfall_dataset = trainer.datamodule.precip_dataset

        files_ = os.listdir(self.save_dir)
        existing_outputs = [f for f in files_ if f.endswith('.pt')]

        # evaluate only on existing outputs
        if self.eval_only_on_existing:
            # Skip the test loop by setting the number of test batches to zero
            yprint(f'\nWARNING: self.eval_only_on_existing is enabled. Skipping test loop. '
                   f'Only evaluating on existing {len(existing_outputs)} batches.\n')

        # check if save_dir already contains outputs
        if not self.skip_existing:
            if existing_outputs:
                raise ValueError(f"Directory {self.save_dir} already contains outputs. "
                                 f"Please remove them or set skip_existing=True.")
        return

    def on_test_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any,
                            batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Skip batch if it already exists.
        """

        if self.eval_only_on_existing:
            pl_module.skip_next_batch = True
            return

        batch_idx = batch[1]['batch_idx']  # for ddp
        if os.path.exists(p.join(self.save_dir, f'batch_{batch_idx}.pt')):
            print(f"Skipping batch {batch_idx}; already exists.")
            pl_module.skip_next_batch = True
            return

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if pl_module.skip_next_batch:
            pl_module.skip_next_batch = False
            return

        batch_dict = outputs['batch_dict']
        batch_idx = batch[1]['batch_idx']  # for ddp
        inverse_norm_batch_dict = self.rainfall_dataset.inverse_normalize_batch(batch_dict)  # tensor
        torch.save(inverse_norm_batch_dict, p.join(self.save_dir, f'batch_{batch_idx}.pt'))
        return

    @rank_zero_only
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        show_metric_for = 'ours'  # FIXME: hard coded for now
        files_ = os.listdir(self.save_dir)
        outputs = [f for f in files_ if f.endswith('.pt')]
        batch_size = torch.load(os.path.join(self.save_dir, outputs[0]))['precip_up'].shape[0]

        print(f"HERE, rank = {pl_module.global_rank}")

        lpips_model = lpips.LPIPS(net='alex').to(pl_module.device)

        gt_hist_total, cpc_inter_hist_total, output_hist_total = None, None, None
        gt_spectra_total, cpc_spectra_total, output_spectral_total = None, None, None

        metrics = []
        for f in tqdm(outputs, desc='Computing metrics'):
            batch_dict = torch.load(p.join(self.save_dir, f))
            cpc_inter_batch = batch_dict['precip_up'].cpu().detach().numpy()
            output_batch = batch_dict['precip_output'].cpu().detach().numpy()
            gt_batch = batch_dict['precip_gt'].cpu().detach().numpy()
            for i in range(batch_size):
                row = {}
                cpc_inter = cpc_inter_batch[i][0, :, :]
                output = output_batch[i][0, :, :]
                gt = gt_batch[i][0, :, :]
                valid_mask = gt != -1
                if not filter_sample_logic(valid_mask, logic='remove_any_invalid'):
                    continue
                # TODO: add noise to gt? that would affect pcc and ssim

                if self.show_vis:
                    vis_sample(cpc_inter, output, gt, self.vis_dir, f, i)

                k = 1
                pooling_func = 'mean'
                # if show_metric_for == 'cpc_inter':
                #     output = cpc_inter  # TODO REMOVE THIS!!!!!

                row['batch'] = f
                row['index'] = i
                row['gt_precip_mean'] = np.mean(gt[valid_mask])
                row['gt_precip_max'] = np.max(gt[valid_mask])

                row['mae'] = calc_mae(output, gt, valid_mask, k=k, pooling_func=pooling_func)
                row['mse'] = calc_mse(output, gt, valid_mask, k=k, pooling_func=pooling_func)
                row['rmse'] = calc_rmse(output, gt, valid_mask, k=k, pooling_func=pooling_func)
                row['pcc'] = calc_pcc(output, gt, valid_mask, k=k, pooling_func=pooling_func)
                # row['ssim'] = calc_ssim(output, gt)
                row['csi'] = calc_csi(output, gt, threshold=10, valid_mask=valid_mask, k=k, pooling_func=pooling_func)
                row['csi_p16'] = calc_csi(output, gt, threshold=10, valid_mask=valid_mask, k=16,
                                          pooling_func=pooling_func)
                row['bias'] = calc_bias(output, gt, valid_mask, k=k, pooling_func=pooling_func)
                row['fss'] = calc_fss(output, gt, threshold=56, k=16)
                row['emd'] = calc_emd(output, gt, valid_mask)
                row['hrre'] = calc_hrre(output, gt, hrre_thres=self.heavy_rain_threshold)
                row['mppe'] = calc_mppe(output, gt, valid_mask)
                row['lpips'] = calc_lpips(output, gt, lpips_model, pl_module.device)
                metrics.append(list(row.values()))

                # TODO: build hist
        metrics_df = pd.DataFrame(metrics, columns=list(row.keys()))

        metrics_df.to_csv(p.join(self.save_dir, f'metrics_{show_metric_for}.csv'))
        # compute summary statistics, mean of each metric, ignore batch and index
        summary_stats_mean = metrics_df.iloc[:, 2:].mean()
        summary_stats_std = metrics_df.iloc[:, 2:].std()
        print(f'----------------Summary Statistics [{show_metric_for}]----------------')
        print("Mean:")
        print(summary_stats_mean)
        print("Standard Deviation:")
        print(summary_stats_std)
        summary_stats = pd.concat([summary_stats_mean, summary_stats_std], axis=1)

        # Name the columns appropriately
        summary_stats.columns = ['Mean', 'Standard Deviation']

        # Save to csv
        summary_stats.to_csv(p.join(self.save_dir, f'summary_stats_{show_metric_for}.csv'))
        return

    def get_ddp_batch_idx(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any,
                            batch_idx: int):
        pass


def filter_sample_logic(valid_mask: np.ndarray, logic: str):
    if logic == 'remove_any_invalid':
        if np.any(~valid_mask):
            return False
    if logic == 'remove_all_invalid':
        if np.all(~valid_mask):
            return False
    return True


def vis_sample(cpc_inter: np.ndarray, output: np.ndarray, gt: np.ndarray, save_dir: str, batch: str, index: int):
    """
    Visualize the sample
    :param cpc_inter:
    :param output:
    :param gt:
    :param save_dir:
    :param batch:
    :param index:
    :return:
    """
    cmap = 'viridis'
    if gt.max() > 0:
        vmax_ = gt.max()
        vmin_ = 0
    else:
        vmax_ = max(cpc_inter.max(), output.max(), gt.max())
        vmin_ = min(cpc_inter.min(), output.min(), gt.min())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im1 = axes[0].imshow(cpc_inter, cmap=cmap, vmin=vmin_, vmax=vmax_)
    axes[0].set_title('CPC Interpolation')
    im2 = axes[1].imshow(output, cmap=cmap, vmin=vmin_, vmax=vmax_)
    axes[1].set_title('Output')
    im3 = axes[2].imshow(gt, cmap=cmap, vmin=vmin_, vmax=vmax_)
    axes[2].set_title('Ground Truth')
    fig.colorbar(im3, ax=axes, pad=0.01)  # add colorbar at the right of the last image, applies to all images
    # plt.show()
    basename = batch.replace('.pt', '')  # This will give '7'
    plt.savefig(p.join(save_dir, f'{basename}_sample_{index}.png'))
    plt.close()
    return


# @monitor_complete
def compute_ensemble_metrics(parent_save_dir, save_dir_pattern, show_vis=False, show_metric_for='ours',
                             ensemble_size=None):
    """
    Static method for meta-analysis
    """
    ensemble_output_path = p.join(parent_save_dir, save_dir_pattern + '_ensemble')
    if not p.exists(ensemble_output_path):
        os.mkdir(ensemble_output_path)
        print(f'Created directory: {ensemble_output_path}')
    runs = os.listdir(parent_save_dir)
    runs = natsorted([r for r in runs if r.startswith(save_dir_pattern) and 'ensemble' not in r])
    runs = natsorted([r for r in runs if r.startswith(save_dir_pattern) and '+' not in r])
    print(f'Found {len(runs)} runs: {runs}')
    if ensemble_size is not None:
        runs = runs[:ensemble_size]
        print(f'Only using the first {ensemble_size} runs')
        print(runs)

    # each run should have the same ckpt
    ckpt_paths = []
    for run in runs:
        save_dir = p.join(parent_save_dir, run)
        with open(p.join(save_dir, 'summary.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'checkpoint' in line:
                    ckpt_paths.append(line.split(': ')[1].strip())
                    break
    assert len(ckpt_paths) == len(runs)
    assert np.all([ckpt_path == ckpt_paths[0] for ckpt_path in ckpt_paths]), 'Checkpoints are not the same'

    # number of common batches in each run, assuming sequential numbering
    num_batches = min([len([b for b in os.listdir(p.join(parent_save_dir, run)) if 'batch_' in b]) for run in runs])
    print(f'Found {num_batches} batches in each run')

    with open(p.join(ensemble_output_path, 'summary.txt'), 'a') as f:
        f.write(f'executed on: {datetime.now()}\n')
        f.write(f'checkpoint: {ckpt_paths[0]}\n')
        f.write(f'ensemble runs: {runs}\n')
        f.write(f'number of batches: {num_batches}\n')

    batch_size = torch.load(os.path.join(parent_save_dir, runs[0], 'batch_0.pt'))['precip_up'].shape[0]

    metrics = []
    for f in tqdm(range(num_batches), desc='Computing ensemble metrics'):

        for i in range(batch_size):
            precip_output, precip_gt = None, None  # numpy array of shape [k, 1, h, w] and [1, h, w]
            # load the output and gt for each run
            for run in runs:
                save_dir = p.join(parent_save_dir, run)
                batch_dict = torch.load(p.join(save_dir, f'batch_{f}.pt'))
                for key, item in batch_dict.items():
                    if key == 'precip_output':
                        cur_precip = item[i, 0, :, :].cpu().detach().numpy()
                        cur_precip = np.expand_dims(cur_precip, axis=0)
                        if precip_output is None:
                            precip_output = cur_precip
                        else:
                            precip_output = np.concatenate([precip_output, cur_precip], axis=0)
                    elif key == 'precip_gt':
                        if precip_gt is None:
                            precip_gt = item[i, 0, :, :].cpu().detach().numpy()
                        else:
                            # check if the gt is the same for all runs
                            assert np.allclose(precip_gt, item[i, 0, :, :].cpu().detach().numpy())
            row = {}
            valid_mask = precip_gt != -1
            if not filter_sample_logic(valid_mask, logic='remove_any_invalid'):
                continue
            precip_output_avg = np.mean(precip_output, axis=0)
            pooling_func = 'mean'
            row['batch'] = f
            row['index'] = i
            row['gt_precip_mean'] = np.mean(precip_gt[valid_mask])
            row['gt_precip_max'] = np.max(precip_gt[valid_mask])
            row['mae'] = calc_mae(precip_output_avg, precip_gt)
            row['crps'] = calc_crps(precip_output, precip_gt)
            row['csi'] = calc_csi(precip_output_avg, precip_gt, threshold=10, valid_mask=valid_mask, k=1, pooling_func=pooling_func)
            row['csi_p16'] = calc_csi(precip_output_avg, precip_gt, threshold=10, valid_mask=valid_mask, k=16, pooling_func=pooling_func)
            row['bias'] = calc_bias(precip_output_avg, precip_gt, valid_mask, k=1, pooling_func=pooling_func)

            metrics.append(list(row.values()))

    metrics_df = pd.DataFrame(metrics, columns=list(row.keys()))
    metrics_df.to_csv(p.join(ensemble_output_path, f'metrics_{ensemble_size}_members.csv'))

    summary_stats_mean = metrics_df.iloc[:, 2:].mean()
    summary_stats_std = metrics_df.iloc[:, 2:].std()
    print(f'----------------Summary Statistics [{show_metric_for}]----------------')
    print("Mean:")
    print(summary_stats_mean)
    print("Standard Deviation:")
    print(summary_stats_std)
    summary_stats = pd.concat([summary_stats_mean, summary_stats_std], axis=1)

    # Name the columns appropriately
    summary_stats.columns = ['Mean', 'Standard Deviation']

    # Save to csv
    summary_stats.to_csv(p.join(ensemble_output_path, f'summary_stats_{ensemble_size}_members.csv'))
    return
