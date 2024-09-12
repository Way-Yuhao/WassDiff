import os
import os.path as p
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import wandb
from matplotlib import pyplot as plt
from lightning.pytorch.callbacks import RichProgressBar, Callback
from rich.progress import Progress
from src.utils.helper import wandb_display_grid, cm_, move_batch_to_cpu, squeeze_batch, extract_values_from_batch


class SaveXarrayResults(Callback):

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir

        # to be defined elsewhere
        self.rainfall_dataset = None
        return

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.rainfall_dataset = trainer.datamodule.precip_dataset

    def on_test_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any,
                            batch_idx: int, dataloader_idx: int = 0) -> None:
        batch_dict, batch_coords, xr_low_res_batch, valid_mask = batch
        rainfall_vis_max = self.get_rainfall_vis_max(xr_low_res_batch)
        self.rainfall_dataset.plot_composite_xarray_batch(xr_low_res_batch, rainfall_vis_max=rainfall_vis_max,
                                                          save_dir=self.save_dir, use_upsampled=False, save_netcdf=True)
        print(f'Low-res Xarray images saved to {self.save_dir}')
        return

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        batch_dict = squeeze_batch(move_batch_to_cpu(outputs['batch_dict']), dim=(0, 1))
        batch_coords = extract_values_from_batch(outputs['batch_coords'])
        xr_low_res_batch = outputs['xr_low_res_batch']
        valid_mask = outputs['valid_mask']

        rainfall_vis_max = self.get_rainfall_vis_max(xr_low_res_batch)
        torch.save(batch, p.join(self.save_dir, 'batch.pt'))
        self.rainfall_dataset.plot_tensor_batch(batch_dict, batch_coords, rainfall_vis_max=rainfall_vis_max,
                                                save_dir=self.save_dir, save_netcdf=True)
        print(f'High-res outputs saved to {self.save_dir}')
        return

    def get_rainfall_vis_max(self, xr_low_res_batch):
        # TODO: implement rainfall_vis_max for historical mode
        return float(np.floor(xr_low_res_batch['precip_gt'].max().values.item()))




