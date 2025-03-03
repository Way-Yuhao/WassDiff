import os
import time
import functools
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, Tuple
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities import rank_zero_only
import wandb
from lightning.pytorch.callbacks import RichProgressBar, Callback, ProgressBar


class GenericLogger(Callback, ABC):
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
        super().__init__()

        self.check_freq_via = check_freq_via
        assert self.check_freq_via in ['global_step', 'epoch']
        self.freqs = {'img': train_log_img_freq, 'score': train_log_score_freq, 'ckpt': train_ckpt_freq}
        self.next_log_idx = {'img': 0 if show_samples_at_start else train_log_img_freq - 1,
                             'score': 0, 'ckpt': train_ckpt_freq - 1}
        self.show_unconditional_samples = show_unconditional_samples
        self.enable_save_ckpt = enable_save_ckpt
        self.add_reference_artifact = add_reference_artifact
        self.report_sample_metrics = report_sample_metrics
        self.sampling_batch_size = sampling_batch_size

        if self.show_unconditional_samples:
            raise NotImplementedError('Unconditional samples not implemented yet.')

        # to be defined elsewhere
        # self.rainfall_dataset = None
        self.progress_bar = None
        self.pbar_type = None
        self.sampling_pbar_desc = 'Sampling on validation set...'
        self.trainer = None
        self.first_batch_visualized = False
        return

    @rank_zero_only
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        wandb.run.summary['logdir'] = trainer.default_root_dir
        self.trainer = trainer
        for callback in trainer.callbacks:
            if isinstance(callback, RichProgressBar):
                self.progress_bar = callback
                self.pbar_type = "rich"
                break
            if isinstance(callback, ProgressBar):
                self.progress_bar = callback
                self.pbar_type = "tqdm"
                break
        return

    @rank_zero_only
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any,
                             batch_idx: int) -> None:
        # visualize the first batch in logger
        if not self.first_batch_visualized:
            self.visualize_batch(**batch)
            self.first_batch_visualized = True
        return

    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any,
                           batch: Any, batch_idx: int) -> None:
        if self._check_frequency(trainer, 'score'):
            self.log_score(pl_module, outputs)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_last_ckpt(trainer)

    @rank_zero_only
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                                batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self._check_frequency(trainer, 'img'):
            self.log_samples(trainer, pl_module, outputs)
        if self._check_frequency(trainer, 'ckpt'):
            self.save_ckpt(trainer)

    @abstractmethod
    def log_score(self, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def log_samples(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def visualize_batch(self, **kwargs):
        pass

    def _check_frequency(self, trainer: "pl.trainer", key: str):
        if self.freqs[key] == -1:
            return False
        if self.check_freq_via == 'global_step':
            check_idx = trainer.global_step
        elif self.check_freq_via == 'epoch':
            check_idx = trainer.current_epoch
        else:
            raise NotImplementedError(f"Frequency check via {self.check_freq_via} not implemented yet.")
        if check_idx >= self.next_log_idx[key]:
            self.next_log_idx[key] = check_idx + self.freqs[key]
            return True
        else:
            return False

    def detect_pbar(self):
        if hasattr(self.progress_bar, "progress") and self.progress_bar.progress is not None:
            self.pbar_type = "rich"
        elif hasattr(self.progress_bar, "set_description"):
            self.pbar_type = "tqdm"
        else:
            raise NotImplementedError("Progress bar type not recognized.")

    def _modify_pbar_desc(self, desc: Optional[str] = None):
        stage = self.trainer.state.stage
        if self.pbar_type == "rich":
            return stage, self._modify_rich_pbar_desc(desc)
        elif self.pbar_type == "tqdm":
            assert stage is not None, "Stage must be provided for tqdm progress bar."
            return stage, self._modify_tqdm_pbar_desc(stage, desc)

    def _revert_pbar_desc(self, stage: Optional[str] = None, progress_info: Optional[Dict] = None):
        if self.pbar_type == "rich":
            self._revert_rich_pbar_desc(progress_info)
        elif self.pbar_type == "tqdm":
            self._revert_tqdm_pbar_desc(stage, progress_info)

    def _modify_tqdm_pbar_desc(self, stage: str, override_desc: str):
        # find the current pbar according to the stage
        if stage == "validate" or stage == "sanity_check":
            current_pbar = self.progress_bar.val_progress_bar
        # elif stage == "train":
        #     current_pbar = self.progress_bar.main_progress_bar
        else:
            raise NotImplementedError()
        original_description = getattr(current_pbar, "desc")
        current_pbar.set_description(override_desc)
        return {"original_description": original_description, "task_id": None}

    def _revert_tqdm_pbar_desc(self, stage: str, progress_info: Optional[Dict] = None):
        original_description = progress_info.get('original_description')
        if stage == "validate" or stage == "sanity_check":
            current_pbar = self.progress_bar.val_progress_bar
        else:
            raise NotImplementedError('Current stage is {}'.format(stage))
        current_pbar.set_description(original_description)

    def _modify_rich_pbar_desc(self, override_desc: str):
        task_id, original_description = None, None
        # Ensure progress bar is active and tasks are initialized
        if self.progress_bar.progress is not None and len(self.progress_bar.progress.tasks) > 0:
            # Look for the current validation task
            for task in self.progress_bar.progress.tasks:
                if "Validation" in task.description or "Sanity Checking" in task.description:
                    task_id = task.id
                    original_description = task.description
                    # Update the description of the active validation progress bar
                    self.progress_bar.progress.update(task_id, description=override_desc)
                    self.progress_bar.progress.refresh()
        return {"original_description": original_description, "task_id": task_id}

    def _revert_rich_pbar_desc(self, progress_info: Optional[Dict] = None):
        task_id, original_description = progress_info.get('task_id'), progress_info.get('original_description')
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
            if self.add_reference_artifact:  # Log the checkpoint as a reference artifact
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


def hold_pbar(desc: Optional[str] = 'Sampling, of course') -> Callable:
    """
    A decorator to update the progress bar description before a long-running
    sampling function executes, and then revert it afterwards.
    The wrapped method is assumed to be an instance method, and the instance
    must implement:
      - _modify_pbar_desc(stage: Optional[str]) -> Tuple[Any, Any]
      - _revert_pbar_desc(task_id, original_description) -> None
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Update the progress bar description
            progress_info: Optional[Dict] = None
            stage: Optional[str] = None
            try:
                stage, progress_info = self._modify_pbar_desc(desc=desc)
                result = func(self, *args, **kwargs) # execute func
            finally:
                # Revert the progress bar description regardless of success/failure.
                if progress_info is not None:
                    self._revert_pbar_desc(stage, progress_info)
            return result
        return wrapper
    return decorator