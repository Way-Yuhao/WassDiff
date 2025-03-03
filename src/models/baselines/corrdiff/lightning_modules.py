from typing import Any, Dict, Tuple, Optional
import torch
from lightning import LightningModule
from modulus.models.diffusion import UNet, EDMPrecondSR
from src.models.precip_downscale import GenericPrecipDownscaleModule
from src.utils.corrdiff_utils.inference import regression_step_only
from src.utils.helper import yprint

class UNetLitModule(GenericPrecipDownscaleModule):

    def __init__(self, net: torch.nn.Module, criterion, model_config, *args, **kwargs):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.model_config = model_config
        self.save_hyperparameters(logger=False, ignore=("net", "optimizer_config", "criterion"))
        return

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            print("Model compiled.")
        elif self.hparams.compile and stage == "test":
            print("Warning: torch_compile is only available during the fit stage.")
        return

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward(self, x, img_lr, sigma, force_fp32=False, **model_kwargs):
        # forcing sigma = 1; assuming model is being used outside of diffusion
        return self.net(x, img_lr=None, sigma=torch.tensor(0.0, device=x.device, dtype=torch.float32))

    def training_step(self, batch: Tuple[Dict, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        batch_dict, _ = batch  # discard coordinates
        condition, gt = self._generate_condition(batch_dict)
        # prediction = self.forward(condition)
        # loss = self.criterion(prediction, gt)
        loss, prediction = self.criterion(net=self.net, img_clean=gt, img_lr=condition)
        loss = loss.mean()
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False, batch_size=condition.shape[0])
        return loss

    def validation_step(self, batch: Tuple[Dict, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        batch_dict, _ = batch
        condition, gt = self._generate_condition(batch_dict)
        # prediction = self.forward(condition)
        # eval_loss = self.criterion(prediction, gt)
        eval_loss, prediction = self.criterion(net=self.net, img_clean=gt, img_lr=condition)
        eval_loss = eval_loss.mean()
        self.log("val/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=False,
                 batch_size=condition.shape[0], sync_dist=True)
        step_output = {"batch_dict": batch_dict, "condition": condition}
        return step_output

    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        latent_shape = (condition.shape[0], 1, condition.shape[2], condition.shape[3])
        image_reg = regression_step_only(net = self.net, img_lr=condition, latents_shape=latent_shape,
                                         lead_time_label=None)
        return image_reg