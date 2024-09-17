"""
Generic Lightning Module for downscaling precipitation data.
"""

from typing import Any, Dict, Tuple, Optional
import wandb
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.ncsn import ncsnpp_cond
from src.models.ncsn.ema import ExponentialMovingAverage
from src.utils.ncsn_utils import sde_lib
from src.utils.ncsn_utils import losses
import src.utils.ncsn_utils.sampling as sampling
from src.models.ncsn import utils as mutils
from src.utils.ncsn_utils import datasets as datasets
# sampling
from src.utils.ncsn_utils.sampling import ReverseDiffusionPredictor, LangevinCorrector
import src.utils.ncsn_utils.controllable_generation as controllable_generation
# debug related imports
from src.utils.ncsn_utils.utils import restore_checkpoint
from src.utils.ncsn_utils.losses import get_optimizer
from src.utils.helper import yprint

__author__ = 'Yuhao Liu'


class GenericPrecipDownscaleModule(LightningModule):

    def __init__(self):
        # TODO