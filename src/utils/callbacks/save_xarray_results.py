import os
import os.path as p
from typing import Any, Dict, Optional
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt
from lightning.pytorch.callbacks import RichProgressBar, Callback
from src.utils.helper import move_batch_to_cpu, squeeze_batch, extract_values_from_batch
from src.utils.metrics import calc_mae, calc_mse, calc_csi, calc_emd
from src.utils.plot_func import plot_psd, plot_distribution, plot_error_map, plot_qq, plot_qq_ensemble_2


class SaveXarrayResults(Callback):
    """
    Callback for specified evaluation targets, saving the low-res and high-res xarray images to disk.
    """

    def __init__(self, save_dir: str, csi_threshold: int, heavy_rain_threshold: int, peak_mesoscale_threshold: int):
        super().__init__()
        self.save_dir = save_dir

        # metric-related constants
        self.csi_threshold = csi_threshold
        self.heavy_rain_threshold = heavy_rain_threshold
        self.peak_mesoscale_threshold = peak_mesoscale_threshold

        # to be defined elsewhere
        self.rainfall_dataset = None
        return

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.rainfall_dataset = trainer.datamodule.precip_dataset

    def on_test_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any,
                            batch_idx: int, dataloader_idx: int = 0) -> None:
        batch_dict, batch_coords, xr_low_res_batch, valid_mask = batch
        rainfall_vis_max = self.get_rainfall_vis_max(xr_low_res_batch)
        self.rainfall_dataset.plot_composite_xarray_batch(xr_low_res_batch, rainfall_vis_max=rainfall_vis_max,
                                                          save_dir=self.save_dir, use_upsampled=False, save_netcdf=True)
        # print(f'Low-res Xarray images saved to {self.save_dir}')
        return

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        batch_dict = squeeze_batch(move_batch_to_cpu(outputs['batch_dict']), dim=(0, 1))
        batch_coords = extract_values_from_batch(outputs['batch_coords'])
        xr_low_res_batch = outputs['xr_low_res_batch']
        valid_mask = outputs['valid_mask']

        rainfall_vis_max = self.get_rainfall_vis_max(xr_low_res_batch)
        torch.save(batch, p.join(self.save_dir, 'batch.pt'))
        # following step rescales output back to mm/day
        self.rainfall_dataset.plot_tensor_batch(batch_dict, batch_coords, rainfall_vis_max=rainfall_vis_max,
                                                save_dir=self.save_dir, save_netcdf=True)
        # print(f'High-res outputs saved to {self.save_dir}')
        self.run_eval_metrics(batch_dict)
        return

    def get_rainfall_vis_max(self, xr_low_res_batch):
        # TODO: implement rainfall_vis_max for historical mode
        return float(np.floor(xr_low_res_batch['precip_gt'].max().values.item()))

    def run_eval_metrics(self, batch: Dict[str, torch.Tensor]):
        ensemble_size = sum(1 for key in batch if key.startswith('precip_output'))
        if ensemble_size == 1:
            self.eval_metrics_single_member(batch)
        else:
            self.eval_metrics_ensemble(batch, ensemble_size)

    def eval_metrics_single_member(self, batch: Dict[str, torch.Tensor]):
        print('Calculating metrics...')
        output = batch['precip_output'].squeeze(0).cpu().numpy()
        gt = batch['precip_gt'].squeeze(0).cpu().numpy()
        # numeric metrics
        mae = calc_mae(output, gt)
        mse = calc_mse(output, gt)
        csi_score = calc_csi(output, gt, threshold=10)
        emd = calc_emd(output.flatten(), gt.flatten())
        print('mae: ', mae)
        print('mse: ', mse)
        print('csi: ', csi_score)
        print('emd: ', emd)

        # figures
        plot_psd(output, gt, self.save_dir)
        plot_distribution(output, gt, self.save_dir, y_log_scale=False)
        plot_distribution(output, gt, self.save_dir, y_log_scale=True)
        plot_error_map(output, gt, self.save_dir)
        plot_qq(output, gt, self.save_dir)

        # save metrics
        with open(p.join(self.save_dir, 'summary.txt'), 'a') as f:
            f.write('mae: ' + str(mae) + '\n')
            f.write('mse: ' + str(mse) + '\n')
            f.write('csi: ' + str(csi_score) + '\n')
            f.write('emd: ' + str(emd) + '\n')
        return

    def eval_metrics_ensemble(self, batch: Dict[str, torch.Tensor], ensemble_size: int):
        print('Calculating metrics for ensemble prediction...')
        gt = batch['precip_gt'].squeeze(0).cpu().numpy()
        # numeric metrics
        maes, mses, csi_scores = [], [], []
        for i in range(ensemble_size):
            print('------sample %d------' % i)
            output = batch['precip_output_' + str(i)].squeeze(0).cpu().numpy()
            maes.append(calc_mae(output, gt))
            mses.append(calc_mse(output, gt))
            csi_scores.append(calc_csi(output, gt, threshold=10))
            print('mae: ', maes[i])
            print('mse: ', mses[i])
            print('csi: ', csi_scores[i])
        print('------ensemble------')
        mae = np.mean(maes)
        mse = np.mean(mses)
        csi_score = np.mean(csi_scores)
        print('avg mae: ', mae)
        print('avg mse: ', mse)
        print('avg csi: ', csi_score)

        # figures
        for i in range(ensemble_size):
            output = batch['precip_output_' + str(i)].squeeze(0).cpu().numpy()
            plot_distribution(output, gt, self.save_dir, y_log_scale=False, suffix=str(i))
            plot_distribution(output, gt, self.save_dir, y_log_scale=True, suffix=str(i))
            plot_error_map(output, gt, self.save_dir, suffix=str(i))
        # plot_psd_ensemble(batch, ensemble_size, self.save_dir)
        plot_qq_ensemble_2(batch, ensemble_size, self.save_dir)

        # save metrics
        with open(p.join(self.save_dir, 'summary.txt'), 'a') as f:
            f.write('------ensemble------\n')
            f.write('avg mae: ' + str(mae) + '\n')
            f.write('avg mse: ' + str(mse) + '\n')
            f.write('avg csi: ' + str(csi_score) + '\n')
            for i in range(ensemble_size):
                f.write('------sample %d------\n' % i)
                f.write('mae: ' + str(maes[i]) + '\n')
                f.write('mse: ' + str(mses[i]) + '\n')
                f.write('csi: ' + str(csi_scores[i]) + '\n')
        return

