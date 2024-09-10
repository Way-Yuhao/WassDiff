import os
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import wandb
from matplotlib import pyplot as plt
from src.utils.helper import wandb_display_grid, cm_


class PrecipDataLogger(Callback):
    def __init__(self, train_log_img_freq: int = 1000, train_log_score_freq: int = 1000,
                 train_log_param_freq: int = 1000, show_samples_at_start: bool = False,
                 show_unconditional_samples: bool = False):
        super().__init__()
        # self.train_log_img_freq = train_log_img_freq
        # self.train_log_score_freq = train_log_score_freq
        # self.train_log_param_freq = train_log_param_freq
        self.freqs = {'img': train_log_img_freq, 'score': train_log_score_freq, 'param': train_log_param_freq}
        self.next_log_idx = {'img': 0 if show_samples_at_start else train_log_img_freq, 'score': 0, 'param': 0}
        # self.show_samples_at_start = show_samples_at_start
        self.show_unconditional_samples = show_unconditional_samples

        if self.show_unconditional_samples:
            raise NotImplementedError('Unconditional samples not implemented yet.')

        # to be defined elsewhere
        self.rainfall_dataset = None
        # self.first_samples_logged = False
        return

    def _check_frequency(self, check_idx: int, key: str):
        if check_idx >= self.next_log_idx[key]:
            self.next_log_idx[key] += self.freqs[key]
            return True
        else:
            return False

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        wandb.run.summary['logdir'] = trainer.default_root_dir
        self.rainfall_dataset = trainer.datamodule.train_val_dataset
        return

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                                batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:

        if self._check_frequency(trainer.global_step, 'img'):
            self._log_samples(pl_module, outputs)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._check_frequency(trainer.global_step, 'score'):
            self._log_score(pl_module, outputs)
        return

    @staticmethod
    def _log_score(pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        s = pl_module.model_config.sampling.sampling_batch_size
        condition = outputs['condition']
        context_mask = outputs['context_mask']
        loss_dict = outputs['loss_dict']
        batch_dict = outputs['batch_dict']
        step = pl_module.global_step
        config = pl_module.model_config
        gt = outputs['batch_dict']['precip_gt']

        if pl_module.model_config.data.condition_mode == 1:
            wandb_display_grid(condition, log_key='train_score/condition', caption='condition', step=step, ncol=s)
        elif config.data.condition_mode in [2, 3]:
            wandb_display_grid(batch_dict['precip_up'], log_key='train_score/condition',
                               caption='low_res upsampled', step=step, ncol=s)
        elif config.data.condition_mode in [4, 5]:
            wandb_display_grid(batch_dict['precip_masked'], log_key='train_score/condition',
                               caption='masked', step=step, ncol=s)
        wandb_display_grid(context_mask, log_key='train_score/mask', caption='context_mask', step=step, ncol=s)
        wandb_display_grid(loss_dict['score'], log_key='train_score/score', caption='score', step=step, ncol=s)
        wandb_display_grid(loss_dict['target'], log_key='train_score/target', caption='target', step=step, ncol=s)
        # wandb_display_grid(loss_dict['noise'], log_key='train_score/noise', caption='noise', step=step, ncol=s)
        wandb_display_grid(loss_dict['perturbed_data'], log_key='train_score/perturbed_data',
                           caption='perturbed_data', step=step, ncol=s)
        wandb_display_grid(loss_dict['denoised_data'], log_key='train_score/denoised_data',
                           caption='denoised_data', step=step, ncol=s)
        wandb_display_grid(loss_dict['error_map'], log_key='train_score/error_map', caption='error_map',
                           step=step, ncol=s)
        wandb_display_grid(gt, log_key='train_score/gt', caption='gt', step=step, ncol=s)
        return

    def _log_samples(self, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        print('Generating conditional samples...')
        condition = outputs['condition']
        batch_dict = outputs['batch_dict']
        gt = outputs['batch_dict']['precip_gt']
        config = pl_module.model_config
        s = pl_module.model_config.sampling.sampling_batch_size
        sampling_null_condition = pl_module.sampling_null_condition
        sample, n = pl_module.sampling_fn(pl_module.net, condition=condition[0:s],
                                          w=config.model.w_guide, null_condition=sampling_null_condition)

        if config.data.condition_mode == 1:
            # display condition and output in one grid
            if config.data.condition_size < config.data.image_size:
                low_res_display = F.interpolate(condition,
                                                size=(config.data.image_size, config.data.image_size),
                                                mode='nearest')
            elif config.data.condition_size == config.data.image_size:
                low_res_display = condition.detach().clone()
        elif config.data.condition_mode in [2, 3, 6]:
            if config.data.condition_size < config.data.image_size:
                low_res_display = F.interpolate(batch_dict['precip_lr'],
                                                size=(config.data.image_size, config.data.image_size),
                                                mode='nearest')
            elif config.data.condition_size == config.data.image_size:
                low_res_display = batch_dict['precip_up'].detach().clone()
        elif config.data.condition_mode in [4, 5]:
            low_res_display = batch_dict['precip_masked'].detach().clone()
        low_res_display = low_res_display.to(config.device)

        low_res_display = self.rainfall_dataset.inverse_normalize_precip(low_res_display)
        sample = self.rainfall_dataset.inverse_normalize_precip(sample)
        gt = self.rainfall_dataset.inverse_normalize_precip(gt)

        concat_samples = torch.cat(
            (low_res_display[0:s, :, :, :], sample[0:s, :, :, :], gt[0:s, :, :, :]), dim=0)
        grid = make_grid(concat_samples, nrow=s)
        grid_mono = grid[0, :, :].unsqueeze(0)
        cm_grid = cm_(grid_mono.detach().cpu(), vmin=0, vmax=max(low_res_display.max(), gt.max()))
        images = wandb.Image(cm_grid, caption='conditional generation [rainfall / output / gt]')
        wandb.log({"val/conditional_samples": images}) #, step=pl_module.global_step)
        self.log_conditional_samples_scaled(low_res_display, sample, gt, n=s,
                                            step=pl_module.global_step, epoch=pl_module.current_epoch)

    @staticmethod
    def log_conditional_samples_scaled(input_row: torch.Tensor, output_row: torch.Tensor, gt_row: torch.Tensor, n: int,
                                       step: Optional[int] = None, epoch: Optional[int] = None):
        input_row = input_row[0:n, :, :, :]
        output_row = output_row[0:n, :, :, :]
        gt_row = gt_row[0:n, :, :, :]
        fig, axs = plt.subplots(3, n, figsize=(n * 5, 15))  # Adjust the figure size as needed

        for i in range(n):
            vmax_ = max(input_row[i].max(), gt_row[i].max())
            # Display input_row images
            im1 = axs[0, i].imshow(input_row[i, 0, :, :].detach().cpu().numpy(), cmap='magma', vmin=0, vmax=vmax_)
            fig.colorbar(im1, ax=axs[0, i], fraction=0.046, pad=0.04)
            axs[0, i].set_xticks([])
            axs[0, i].set_yticks([])
            # Display output_row images
            im2 = axs[1, i].imshow(output_row[i, 0, :, :].detach().cpu().numpy(), cmap='magma', vmin=0, vmax=vmax_)
            fig.colorbar(im2, ax=axs[1, i], fraction=0.046, pad=0.04)
            axs[1, i].set_xticks([])
            axs[1, i].set_yticks([])
            # Display gt_row images
            im3 = axs[2, i].imshow(gt_row[i, 0, :, :].detach().cpu().numpy(), cmap='magma', vmin=0, vmax=vmax_)
            fig.colorbar(im3, ax=axs[2, i], fraction=0.046, pad=0.04)
            axs[2, i].set_xticks([])
            axs[2, i].set_yticks([])
        plt.tight_layout()
        plt.close(fig)
        # Log the figure to wandb
        images = wandb.Image(fig)
        wandb.log({"val/conditional_samples_scaled": images})
        return
