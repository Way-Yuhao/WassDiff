from typing import Any, Dict, Tuple, Optional
import torch
from lightning import LightningModule
from modulus.models.diffusion import UNet, EDMPrecondSR
from modulus.metrics.diffusion import RegressionLoss, ResLoss, RegressionLossCE
from src.utils.helper import yprint

class UNetLitModule(LightningModule):
    def __init__(self, net: torch.nn.Module, *args, **kwargs):
        super().__init__()
        self.net = net

