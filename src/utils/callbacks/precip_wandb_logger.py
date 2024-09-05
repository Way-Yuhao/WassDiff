import os
from typing import Any

import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT


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

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        print('global step = ', pl_module.global_step)
