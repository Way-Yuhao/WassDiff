"""
Ablation models for downscaling precipitation
"""
from typing import Any, Dict, Tuple, Optional, List, Union
import torch
from src.models.precip_downscale.wassdiff import WassDiffLitModule
from src.utils.helper import yprint

__author__ = 'Yuhao Liu'


class WassDiffAblationERA5Module(WassDiffLitModule):

    def __init__(self, model_config: dict, optimizer_config: dict,
                 compile: bool, num_samples: int = 1, pytorch_ckpt_path: Optional[str] = None,
                 exclude_var: Optional[Union[str, List[str]]] = None, *args, **kwargs) -> None:

        # dynamically modify config file
        if exclude_var is None:
            raise ValueError('exclude_var must be provided')
        elif isinstance(exclude_var, str):
            exclude_var = [exclude_var]
        self.exclude_var = exclude_var
        model_config.data.num_context_chs -= len(exclude_var)
        yprint(f'Initializing ablation model with exclude_var: {exclude_var}')

        super().__init__(model_config, optimizer_config, compile, num_samples, pytorch_ckpt_path)
        return

    def _generate_condition(self, batch_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        assert self.model_config.data.condition_mode in [2, 6], 'Ablation model only supports condition_mode 2 or 6'

        # check if keys to exclude are present in batch_dict
        if not all(var in batch_dict for var in self.exclude_var):
            # At least one variable is missing
            missing_vars = [var for var in self.exclude_var if var not in batch_dict]
            raise AssertionError(f"The following variables are missing from batch_dict: {missing_vars}")

        y = self.scaler(batch_dict['precip_gt'])
        exclude_keys = ['precip_lr', 'precip_gt']
        exclude_keys.extend(self.exclude_var)  # ABLATION: exclude the variables in exclude_var
        tensors_to_stack = [tensor for key, tensor in batch_dict.items() if key not in exclude_keys]
        stacked_tensor = torch.cat(tensors_to_stack, dim=1)
        condition = stacked_tensor

        return condition, y
