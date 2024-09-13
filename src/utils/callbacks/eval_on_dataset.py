import os
import os.path as p
from typing import Any, Dict, Optional
import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
import wandb
from matplotlib import pyplot as plt
from lightning.pytorch.callbacks import RichProgressBar, Callback


class EvalOnDataset(Callback):
    """
    Callback for evaluating on a specified dataset.
    """

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir

        # to be defined elsewhere
        self.rainfall_dataset = None
        return

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.rainfall_dataset = trainer.datamodule.precip_dataset

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        batch_dict = outputs['batch_dict']
        inverse_norm_batch_dict = self.rainfall_dataset.inverse_normalize_batch(batch_dict)  # tensor
        torch.save(inverse_norm_batch_dict, p.join(self.save_dir, f'batch_{batch_idx}.pt'))
        return

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass  # TODO: implement eval metrics

