"""
Ablation models for downscaling precipitation
"""
from typing import Any, Dict, Tuple, Optional
import torch
from src.models.precip_downscale.wassdiff import WassDiffLitModule

__author__ = 'Yuhao Liu'


class WassDiffAblationERA5Module(WassDiffLitModule):

    def __init__(self, model_config: dict, optimizer_config: dict,
                 compile: bool, num_samples: int = 1, pytorch_ckpt_path: Optional[str] = None,
                 exclude_var: Optional[str] = None,) -> None:

        super().__init__(model_config, optimizer_config, compile, num_samples, pytorch_ckpt_path)
        self.exclude_var = exclude_var
        print('EXCLUDING VAR', exclude_var)

