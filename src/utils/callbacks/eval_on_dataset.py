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
from src.utils.helper import yprint


class EvalOnDataset(Callback):
    """
    Callback for evaluating on a specified dataset.
    """

    def __init__(self, save_dir: str, skip_existing: bool, eval_only_on_existing: bool):
        super().__init__()
        self.save_dir = save_dir
        self.skip_existing = skip_existing
        self.eval_only_on_existing = eval_only_on_existing

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
        elif os.path.exists(p.join(self.save_dir, f'batch_{batch_idx}.pt')):
            print(f"Skipping batch {batch_idx}")
            pl_module.skip_next_batch = True
            return

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if pl_module.skip_next_batch:
            pl_module.skip_next_batch = False
            return

        batch_dict = outputs['batch_dict']
        inverse_norm_batch_dict = self.rainfall_dataset.inverse_normalize_batch(batch_dict)  # tensor
        torch.save(inverse_norm_batch_dict, p.join(self.save_dir, f'batch_{batch_idx}.pt'))
        return

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('HEREEE')

