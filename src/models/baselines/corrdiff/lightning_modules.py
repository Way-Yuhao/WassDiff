from typing import Any, Dict, Tuple, Optional
import torch
from src.models.precip_downscale import GenericPrecipDownscaleModule
from src.utils.corrdiff_utils.inference import regression_step_only, diffusion_step_batch
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

    def test_step(self, batch: Tuple[Dict, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        if self.skip_next_batch:  # determine whether to skip this batch
            return {}
        batch_dict, batch_coords, xr_low_res_batch, valid_mask = batch
        condition, gt = self._generate_condition(batch_dict)
        # ensemble prediction, if needed
        if self.hparams.num_samples > 1 and condition.shape[0] > 1:
            raise AttributeError('Ensemble prediction not supported for batch size > 1.')
        elif self.hparams.num_samples > 1 and condition.shape[0] == 1:
            condition = condition.repeat(self.hparams.num_samples, 1, 1, 1)


        latent_shape = (condition.shape[0], 1, condition.shape[2], condition.shape[3])
        x = regression_step_only(net=self.net, img_lr=condition, latents_shape=latent_shape,
                                         lead_time_label=None)
        if self.hparams.num_samples == 1:
            batch_dict['precip_output'] = x
        else:
            for i in range(self.hparams.num_samples):
                batch_dict['precip_output_' + str(i)] = x[i, :, :, :]

        return {'batch_dict': batch_dict, 'batch_coords': batch_coords, 'xr_low_res_batch': xr_low_res_batch,
                'valid_mask': valid_mask}

    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        latent_shape = (condition.shape[0], 1, condition.shape[2], condition.shape[3])
        image_reg = regression_step_only(net = self.net, img_lr=condition, latents_shape=latent_shape,
                                         lead_time_label=None)
        return image_reg


class CorrDiffLiTModule(GenericPrecipDownscaleModule):

    def __init__(self, net: torch.nn.Module, criterion, model_config, sampling, *args, **kwargs):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.sampler = sampling
        self.model_config = model_config
        self.save_hyperparameters(logger=False, ignore=("net", "optimizer_config", "criterion"))

        # to be defined elsewhere
        self.regression_net = None
        return

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            print("Model compiled.")
        elif self.hparams.compile and stage == "test":
            print("Warning: torch_compile is only available during the fit stage.")
        self.load_regression_net()
        self.criterion = self.criterion(regression_net=self.regression_net)
        return

    def load_regression_net(self):
        """
        Loads from .ckpt
        """
        ckpt_ = self.hparams.regression_net_ckpt
        self.regression_net = torch.load(ckpt_, map_location=self.device)
        yprint(f"Regression net loaded from {ckpt_}")
        return

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # forward(self, x, img_lr, sigma, force_fp32=False, **model_kwargs):
    #     # forcing sigma = 1; assuming model is being used outside of diffusion
    #     return self.net(x, img_lr=None, sigma=torch.tensor(0.0, device=x.device, dtype=torch.float32))
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

    def test_step(self, batch: Tuple[Dict, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        if self.skip_next_batch:  # determine whether to skip this batch
            return {}
        batch_dict, batch_coords, xr_low_res_batch, valid_mask = batch
        condition, gt = self._generate_condition(batch_dict)
        # ensemble prediction, if needed
        if self.hparams.num_samples > 1 and condition.shape[0] > 1:
            raise AttributeError('Ensemble prediction not supported for batch size > 1.')
        elif self.hparams.num_samples > 1 and condition.shape[0] == 1:
            condition = condition.repeat(self.hparams.num_samples, 1, 1, 1)

        latent_shape = (condition.shape[0], 1, condition.shape[2], condition.shape[3])
        prediction_mean = regression_step_only(net=self.regression_net, img_lr=condition, latents_shape=latent_shape,
                                               lead_time_label=None)
        prediction_residual = diffusion_step_batch(
            net=self.net,
            sampler_fn=self.sampler,
            img_lr=condition,
            img_shape=condition.shape[2:],
            img_out_channels=1,
            device=self.device,
            hr_mean=None,
            lead_time_label=None,
        )
        x = prediction_mean + prediction_residual
        if self.hparams.num_samples == 1:
            batch_dict['precip_output'] = x
        else:
            for i in range(self.hparams.num_samples):
                batch_dict['precip_output_' + str(i)] = x[i, :, :, :]

        return {'batch_dict': batch_dict, 'batch_coords': batch_coords, 'xr_low_res_batch': xr_low_res_batch,
                'valid_mask': valid_mask}

    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        latent_shape = (condition.shape[0], 1, condition.shape[2], condition.shape[3])
        prediction_mean = regression_step_only(net=self.regression_net, img_lr=condition, latents_shape=latent_shape,
                                               lead_time_label=None)
        prediction_residual = diffusion_step_batch(
                net=self.net,
                sampler_fn=self.sampler,
                img_lr=condition,
                img_shape=condition.shape[2:],
                img_out_channels=1,
                device=self.device,
                hr_mean=None,
                lead_time_label=None,
        )
        prediction = prediction_mean + prediction_residual
        return prediction