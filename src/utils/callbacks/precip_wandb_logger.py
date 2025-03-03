import os
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities import rank_zero_only
import wandb
from matplotlib import pyplot as plt
from lightning.pytorch.callbacks import RichProgressBar, Callback
from src.utils.callbacks.generic_wandb_logger import GenericLogger, hold_pbar
from src.utils.helper import wandb_display_grid, cm_
from src.utils.helper import visualize_batch as visualize_batch_static
from src.utils.metrics import calc_mae, calc_bias


class PrecipDataLogger(GenericLogger):
    def __init__(self, train_log_img_freq: int = 1000, train_log_score_freq: int = 1000,
                 train_ckpt_freq: int = 1000, show_samples_at_start: bool = False,
                 show_unconditional_samples: bool = False, check_freq_via: str = 'global_step',
                 enable_save_ckpt: bool = False, add_reference_artifact: bool = False,
                 report_sample_metrics: bool = True, sampling_batch_size: int = 6):
        """
        Callback to log images, scores and parameters to wandb.
        :param train_log_img_freq: frequency to log images. Set to -1 to disable
        :param train_log_score_freq: frequency to log scores. Set to -1 to disable
        :param train_ckpt_freq: frequency to log parameters. Set to -1 to disable
        :param show_samples_at_start: whether to log samples at the start of training (likely during sanity check)
        :param show_unconditional_samples: whether to log unconditional samples. Deprecated.
        :param check_freq_via: whether to check frequency via 'global_step' or 'epoch'
        :param enable_save_ckpt: whether to save checkpoint
        :param add_reference_artifact: whether to add the checkpoint as a reference artifact
        :param report_sample_metrics: whether to report sample metrics
        :param sampling_batch_size: number of samples to visualize
        """
        super().__init__(train_log_img_freq, train_log_score_freq, train_ckpt_freq, show_samples_at_start,
                         show_unconditional_samples, check_freq_via, enable_save_ckpt, add_reference_artifact,
                         report_sample_metrics, sampling_batch_size)
        # additional attributes specific for rainfall
        self.rainfall_dataset = None

    @rank_zero_only
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_fit_start(trainer, pl_module)
        self.rainfall_dataset = trainer.datamodule.precip_dataset

    @rank_zero_only
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any,
                             batch_idx: int) -> None:
        # visualize the first batch in logger
        batch, _ = batch  # discard coordinates
        if not self.first_batch_visualized:
            visualize_batch_static(**batch)
            self.first_batch_visualized = True
        return

    def log_score(self, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
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

    @hold_pbar("sampling...")
    def log_samples(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        condition = outputs['condition']
        batch_dict = outputs['batch_dict']
        gt = outputs['batch_dict']['precip_gt']
        config = pl_module.model_config
        s = self.sampling_batch_size

        # sample, n = pl_module.sampling_fn(pl_module.net, condition=condition[0:s],
        #                                   w=config.model.w_guide, null_condition=sampling_null_condition)
        sample = pl_module.sample(condition[0:s])

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
        low_res_display = low_res_display.to(pl_module.device)

        low_res_display = self.rainfall_dataset.inverse_normalize_precip(low_res_display)
        sample = self.rainfall_dataset.inverse_normalize_precip(sample)
        gt = self.rainfall_dataset.inverse_normalize_precip(gt)

        self.log_conditional_samples_scaled(low_res_display, sample, gt, n=s)

        # log sample metrics
        if self.report_sample_metrics:
            mae = calc_mae(sample.cpu().detach().numpy(), gt[0:s, :, :, :].cpu().detach().numpy(), valid_mask=None, k=1,
                           pooling_func='mean')
            bias = calc_bias(sample.cpu().detach().numpy(), gt[0:s, :, :, :].cpu().detach().numpy(), valid_mask=None,
                             k=1,
                             pooling_func='mean')
            wandb.log({'val/sample_mae': mae, 'val/sample_bias': bias, 'epoch': trainer.current_epoch})

    @hold_pbar("sampling...")
    def log_conditional_samples_scaled(self, input_row: torch.Tensor, output_row: torch.Tensor, gt_row: torch.Tensor, n: int,
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

    def visualize_batch(self, **kwargs):
        return visualize_batch_static(**kwargs)


class LegacyPrecipDataLogger(Callback):
    def __init__(self, train_log_img_freq: int = 1000, train_log_score_freq: int = 1000,
                 train_log_param_freq: int = 1000, show_samples_at_start: bool = False,
                 show_unconditional_samples: bool = False, check_freq_via: str = 'global_step',
                 enable_save_ckpt: bool = False, add_reference_artifact: bool = False,
                 report_sample_metrics: bool = True):
        """
        Callback to log images, scores and parameters to wandb.
        :param train_log_img_freq: frequency to log images. Set to -1 to disable
        :param train_log_score_freq: frequency to log scores. Set to -1 to disable
        :param train_log_param_freq: frequency to log parameters. Set to -1 to disable
        :param show_samples_at_start: whether to log samples at the start of training (likely during sanity check)
        :param show_unconditional_samples: whether to log unconditional samples. Deprecated.
        :param check_freq_via: whether to check frequency via 'global_step' or 'epoch'
        :param enable_save_ckpt: whether to save checkpoint
        :param add_reference_artifact: whether to add the checkpoint as a reference artifact
        :param report_sample_metrics: whether to report sample metrics
        """
        super().__init__()

        self.check_freq_via = check_freq_via
        assert self.check_freq_via in ['global_step', 'epoch']
        self.freqs = {'img': train_log_img_freq, 'score': train_log_score_freq, 'param': train_log_param_freq}
        self.next_log_idx = {'img': 0 if show_samples_at_start else train_log_img_freq - 1, 'score': 0, 'param': 0}
        self.show_unconditional_samples = show_unconditional_samples
        self.enable_save_ckpt = enable_save_ckpt
        self.add_reference_artifact = add_reference_artifact
        self.report_sample_metrics = report_sample_metrics

        if self.show_unconditional_samples:
            raise NotImplementedError('Unconditional samples not implemented yet.')

        # to be defined elsewhere
        self.rainfall_dataset = None
        self.progress_bar = None
        self.sampling_pbar_desc = 'Sampling on validation set...'
        # self.first_samples_logged = False
        self.first_batch_visualized = False
        return

    @rank_zero_only
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        wandb.run.summary['logdir'] = trainer.default_root_dir
        self.rainfall_dataset = trainer.datamodule.precip_dataset

        for callback in trainer.callbacks:
            if isinstance(callback, RichProgressBar):
                self.progress_bar = callback
                break
        return

    @rank_zero_only
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any,
                             batch_idx: int) -> None:
        # visualize the first batch in logger
        batch, _ = batch  # discard coordinates
        if not self.first_batch_visualized:
            visualize_batch_static(**batch)
            self.first_batch_visualized = True
        return

    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any,
                           batch: Any, batch_idx: int) -> None:
        if self._check_frequency(trainer, 'score'):
            self._log_score(pl_module, outputs)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_last_ckpt(trainer)

    @rank_zero_only
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                                batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self._check_frequency(trainer, 'img'):
            self._log_samples(trainer, pl_module, outputs)
            self.save_ckpt(trainer)

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

    def _log_samples(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        pbar_taskid, original_pbar_desc = self._modify_pbar_desc()

        condition = outputs['condition']
        batch_dict = outputs['batch_dict']
        gt = outputs['batch_dict']['precip_gt']
        config = pl_module.model_config
        s = pl_module.model_config.sampling.sampling_batch_size

        # sample, n = pl_module.sampling_fn(pl_module.net, condition=condition[0:s],
        #                                   w=config.model.w_guide, null_condition=sampling_null_condition)
        sample = pl_module.sample(condition[0:s])

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
        low_res_display = low_res_display.to(pl_module.device)

        low_res_display = self.rainfall_dataset.inverse_normalize_precip(low_res_display)
        sample = self.rainfall_dataset.inverse_normalize_precip(sample)
        gt = self.rainfall_dataset.inverse_normalize_precip(gt)

        self.log_conditional_samples_scaled(low_res_display, sample, gt, n=s)

        # log sample metrics
        if self.report_sample_metrics:
            mae =  calc_mae(sample.cpu().detach().numpy(), gt[0:s, :, :, :].cpu().detach().numpy(), valid_mask=None, k=1,
                            pooling_func='mean')
            bias = calc_bias(sample.cpu().detach().numpy(), gt[0:s, :, :, :].cpu().detach().numpy(), valid_mask=None, k=1,
                             pooling_func='mean')
            wandb.log({'val/sample_mae': mae, 'val/sample_bias': bias, 'epoch': trainer.current_epoch})
        
        self._revert_pbar_desc(pbar_taskid, original_pbar_desc)
        return

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

    def _check_frequency(self, trainer: "pl.trainer", key: str):
        if self.freqs[key] == -1:
            return False
        if self.check_freq_via == 'global_step':
            check_idx = trainer.global_step
        elif self.check_freq_via == 'epoch':
            check_idx = trainer.current_epoch
        if check_idx >= self.next_log_idx[key]:
            self.next_log_idx[key] = check_idx + self.freqs[key]
            return True
        else:
            return False

    def _modify_pbar_desc(self):
        task_id, original_description = None, None
        # Ensure progress bar is active and tasks are initialized
        if self.progress_bar.progress is not None and len(self.progress_bar.progress.tasks) > 0:
            # Look for the current validation task
            for task in self.progress_bar.progress.tasks:
                if "Validation" in task.description or "Sanity Checking" in task.description:
                    task_id = task.id
                    original_description = task.description
                    # Update the description of the active validation progress bar
                    self.progress_bar.progress.update(task_id, description=self.sampling_pbar_desc)
                    self.progress_bar.progress.refresh()
        return task_id, original_description

    def _revert_pbar_desc(self, task_id, original_description):
        # Ensure progress bar is active and tasks are initialized
        if self.progress_bar.progress is not None and len(self.progress_bar.progress.tasks) > 0:
            # Look for the current validation task
            for task in self.progress_bar.progress.tasks:
                if task.id == task_id:
                    # Update the description of the active validation progress bar
                    self.progress_bar.progress.update(task_id, description=original_description)
                    self.progress_bar.progress.refresh()
        return


    def save_ckpt(self, trainer: Trainer):
        if self.enable_save_ckpt:
            save_dir = trainer.logger.save_dir
            ckpt_path = os.path.join(save_dir, 'checkpoints',
                                     f'epoch_{trainer.current_epoch:03d}_step_{wandb.run.step:03d}.ckpt')
            trainer.save_checkpoint(ckpt_path)
            if self.add_reference_artifact: # Log the checkpoint as a reference artifact
                artifact = wandb.Artifact(name=f'model-ckpt-{wandb.run.id}', type='model')
                artifact.add_reference(f"file://{ckpt_path}")
                artifact.metadata = {'epoch': trainer.current_epoch, 'step': wandb.run.step}
                wandb.run.log_artifact(artifact, aliases=[f'epoch_{trainer.current_epoch:03d}',
                                                          f'step_{wandb.run.step:03d}'])
        return

    def save_last_ckpt(self, trainer: Trainer):
        if self.enable_save_ckpt:
            save_dir = trainer.logger.save_dir
            ckpt_path = os.path.join(save_dir, 'checkpoints', 'last.ckpt')
            trainer.save_checkpoint(ckpt_path)

    @staticmethod
    def remove_image_file_from_summary():
        # FIXME: this is not working...
        for key in list(wandb.run.summary.keys()):
            value = wandb.run.summary[key]
            if isinstance(value, dict) and value.get('_type') == 'image-file':
                del wandb.run.summary[key]
        return