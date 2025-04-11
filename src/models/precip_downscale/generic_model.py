"""
Generic Lightning Module for downscaling precipitation data.
"""

from typing import Dict, Tuple
from abc import ABC, abstractmethod
import torch
from lightning import LightningModule


__author__ = 'Yuhao Liu'


class GenericPrecipDownscaleModule(LightningModule, ABC):

    def __init__(self):
        super().__init__()
        # internal flags
        self.first_batch_visualized = False
        self.skip_next_batch = False  # flag to be modified by callbacks


    def _generate_condition(self, batch_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parses input batch dictionary and returns the condition and target tensors.
        """
        y = batch_dict['precip_gt']

        if self.model_config.data.condition_mode == 0:
            raise AttributeError()  # deprecated
        elif self.model_config.data.condition_mode == 1:
            condition = batch_dict['precip_up']
        elif self.model_config.data.condition_mode in [2, 6]:
            exclude_keys = ['precip_lr', 'precip_gt']
            tensors_to_stack = [tensor for key, tensor in batch_dict.items() if key not in exclude_keys]
            stacked_tensor = torch.cat(tensors_to_stack, dim=1)
            condition = stacked_tensor
        elif self.model_config.data.condition_mode == 3:
            exclude_keys = ['precip_lr', 'precip_gt', 'precip_up']
            tensors_to_stack = [tensor for key, tensor in batch_dict.items() if key not in exclude_keys]
            stacked_tensor = torch.cat(tensors_to_stack, dim=1)
            condition = stacked_tensor
        elif self.model_config.data.condition_mode == 4:
            condition = batch_dict['precip_masked']
        elif self.model_config.data.condition_mode == 5:
            exclude_keys = ['precip_gt', 'mask']
            tensors_to_stack = [tensor for key, tensor in batch_dict.items() if key not in exclude_keys]
            stacked_tensor = torch.cat(tensors_to_stack, dim=1)
            condition = stacked_tensor
        else:
            raise AttributeError()
        return condition, y

    @abstractmethod
    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Runs sampling fn when given condition tensor. No further trimming or processing is performance.
        This is intended to use during validation to gather quick samples.
        """
        pass
