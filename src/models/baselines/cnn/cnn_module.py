from typing import Any, Dict, Tuple, Optional
import torch
from src.models.precip_downscale import GenericPrecipDownscaleModule
from src.utils.helper import yprint


class CNNLitModule(GenericPrecipDownscaleModule):
    """
    A PyTorch Lightning module for a simple CNN.
    Serves to provide a deterministic baseline for comparison.
    Inherits from `GenericE2DModule` which provides a template for event-to-depth models.
    """

    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 criterion, compile: bool = False,
                 *args, **kwargs) -> None:
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.save_hyperparameters(logger=False, ignore=("net", "optimizer_config", "criterion"))
        return

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            print("Model compiled.")
        elif self.hparams.compile and stage == "test":
            print("Warning: torch_compile is only available during the fit stage.")
        # if self.hparams.allow_resize:
        #     yprint("Resizing data to multiples of 16 to be compatible with UNet.")
        return

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        batch_dict, _ = batch  # discard coordinates
        condition, gt = self._generate_condition(batch)
        condition, gt = self.resize_data(condition, gt)
        prediction = self.forward(condition)
        loss = self.criterion(prediction, gt)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False, batch_size=condition.shape[0])
        # step_output = {"batch_dict": batch, "loss_dict": loss_dict, 'condition': condition, 'context_mask': context_mask}
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step on a batch of data from the validation set.
        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        batch_dict, _ = batch
        condition, gt = self._generate_condition(batch)
        condition, gt = self.resize_data(condition, gt)
        prediction = self.forward(condition)
        eval_loss = self.criterion(prediction, gt)
        self.log("val/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=False,
                 batch_size=condition.shape[0], sync_dist=True)
        step_output = {"batch_dict": batch, "condition": condition}
        return step_output

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        if self.skip_next_batch:  # determine whether to skip this batch
            return {}
        batch_dict, batch_coords, xr_low_res_batch, valid_mask = batch
        condition, gt = self._generate_condition(batch_dict)
        x = self.forward(condition)
        if self.hparams.num_samples == 1:
            batch_dict['precip_output'] = x
        else:
            for i in range(self.hparams.num_samples):
                batch_dict['precip_output_' + str(i)] = x[i, :, :, :]

        return {'batch_dict': batch_dict, 'batch_coords': batch_coords, 'xr_low_res_batch': xr_low_res_batch,
                'valid_mask': valid_mask}

    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        return self.forward(condition)

