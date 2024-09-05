import os
from typing import Any, Dict
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import wandb
from src.utils.helper import wandb_display_grid


class PrecipDataLogger(Callback):
    def __init__(self, train_log_img_freq: int = 1000, train_log_score_freq: int = 1000,
                 train_log_param_freq: int = 1000, show_samples_at_start: bool = False,
                 show_unconditional_samples: bool = False):
        super().__init__()
        self.train_log_img_freq = train_log_img_freq
        self.train_log_score_freq = train_log_score_freq
        self.train_log_param_freq = train_log_param_freq
        self.show_samples_at_start = show_samples_at_start
        self.show_unconditional_samples = show_unconditional_samples
        return

    def _check_frequency(self, check_idx):
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        print(0)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        if trainer.global_step % self.train_log_score_freq == 0:
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

    def _log_samples(self, pl_module: LightningModule, batch: Dict[str, torch.Tensor]):
        pass