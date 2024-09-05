from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.ncsn import ncsnpp_cond
from src.models.ncsn.ema import ExponentialMovingAverage
from src.utils.ncsn_utils import sde_lib
from src.utils.ncsn_utils import losses
from src.utils.helper import visualize_batch
import src.utils.ncsn_utils.sampling as sampling
from src.models.ncsn import utils as mutils
from src.utils.ncsn_utils import datasets as datasets


class WassDiffLitModule(LightningModule):
    """
    """

    def __init__(self,
                 # optimizer: torch.optim.Optimizer,
                 # scheduler: torch.optim.lr_scheduler,
                 compile: bool,
                 model_config: dict,
                 optimizer_config: dict,
                 ) -> None:
        """Initialize a `WassDiffLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        self.net = mutils.create_model(model_config)
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=("model_config", "optimizer_config"))
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # TODO defined loss function and metrics

        # attributes to be defined elsewhere
        self.train_step_fn = None
        self.eval_step_fn = None

        # internal flags
        self.automatic_optimization = False  # TODO verify
        self.first_batch_visualized = False
        return

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # Compile via JiT
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
        return

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        self.model_config.device = self.device # TODO: verify

        # Create data normalizer and its inverse
        self.scaler = datasets.get_data_scaler(self.model_config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.model_config)

        optimizer = losses.get_optimizer(self.optimizer_config, self.net.parameters())

        ema = ExponentialMovingAverage(self.net.parameters(), decay=self.model_config.model.ema_rate)
        self.state = dict(optimizer=optimizer, model=self.net, ema=ema, step=0)

        # Setup SDEs
        if self.model_config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=self.model_config.model.beta_min, beta_max=self.model_config.model.beta_max,
                                N=self.model_config.model.num_scales)
            sampling_eps = 1e-3
        elif self.model_config.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=self.model_config.model.beta_min, beta_max=self.model_config.model.beta_max,
                                   N=self.model_config.model.num_scales)
            sampling_eps = 1e-3
        elif self.model_config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=self.model_config.model.sigma_min,
                                sigma_max=self.model_config.model.sigma_max,
                                N=self.model_config.model.num_scales)
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {self.hparams.sde} unknown.")

        # Build one-step training and evaluation functions
        optimize_fn = losses.optimization_manager(self.optimizer_config)
        continuous = self.model_config.training.continuous
        reduce_mean = self.model_config.training.reduce_mean
        likelihood_weighting = self.model_config.training.likelihood_weighting
        self.use_emd = self.model_config.training.use_emd
        emd_weight = self.model_config.training.emd_weight
        self.train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                                reduce_mean=reduce_mean, continuous=continuous,
                                                likelihood_weighting=likelihood_weighting,
                                                use_emd=self.use_emd, emd_weight=emd_weight,
                                                compute_rescaled_emd=self.model_config.training.compute_rescaled_emd,
                                                emd_rescale_c=self.model_config.data.precip_rescale_c)

        self.eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                               reduce_mean=reduce_mean, continuous=continuous,
                                               likelihood_weighting=likelihood_weighting,
                                               use_emd=self.use_emd, emd_weight=emd_weight,
                                               compute_rescaled_emd=self.model_config.training.compute_rescaled_emd,
                                               emd_rescale_c=self.model_config.data.precip_rescale_c)

        # Building sampling functions
        sampling_shape = (self.model_config.sampling.sampling_batch_size, self.model_config.data.num_channels,
                          self.model_config.data.image_size, self.model_config.data.image_size)
        # TODO: verify cuda device
        sampling_fn = sampling.get_sampling_fn(self.model_config, sde, sampling_shape, self.inverse_scaler,
                                               sampling_eps)
        s = self.model_config.sampling.sampling_batch_size
        num_train_steps = self.model_config.training.n_iters

        # if self.hparams.scheduler is not None:
        #     scheduler = self.hparams.scheduler(optimizer=optimizer)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }
        # return {"optimizer": optimizer}
        return optimizer

    def on_fit_start(self) -> None:
        """Lightning hook that is called when the fit begins."""

        self.sampling_null_condition = self._generate_null_condition()
        self.state['optimizer'] = self.optimizers()  # TODO: verify

    def on_train_batch_start(self, batch, batch_idx) -> None:
        batch, _ = batch  # discard coordinates
        # visualize the first batch in logger
        if not self.first_batch_visualized:
            visualize_batch(**batch)
            self.first_batch_visualized = True
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of images.
        """
        return self.net(x)

    def model_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        raise NotImplementedError()

    def training_step(
            self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        batch, _ = batch  # discard coordinates
        condition, batch = self._generate_condition(batch)
        condition = self._dropout_condition(condition)
        loss, loss_dict = self.train_step_fn(self.state, batch, condition)

        # TODO log freq?
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.use_emd:
            self.log("train/emd_loss", loss_dict['emd_loss'], on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/score_loss", loss_dict['score_loss'], on_step=False, on_epoch=True, prog_bar=True)
        return loss  # TODO: verify

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch, _ = batch  # discard coordinates
        condition, batch = self._generate_condition(batch)
        eval_loss, _ = self.eval_step_fn(self.state, batch, condition)
        self.log("val/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=True)
        return

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        raise NotImplementedError()

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def _generate_null_condition(self):
        # generate null condition for sampling
        s = self.model_config.sampling.sampling_batch_size
        if self.model_config.data.condition_mode in [1, 4]:
            sampling_null_condition = torch.ones((s, self.model_config.data.num_channels, self.model_config.data.condition_size,
                                                  self.model_config.data.condition_size)) * self.model_config.model.null_token
        elif self.model_config.data.condition_mode in [2, 5, 6]:
            sampling_null_condition = torch.ones((s, self.model_config.data.num_channels + self.model_config.data.num_context_chs,
                                                  self.model_config.data.image_size,
                                                  self.model_config.data.image_size)) * self.model_config.model.null_token
        elif self.model_config.data.condition_mode == 3:
            sampling_null_condition = torch.ones((s, self.model_config.data.num_context_chs, self.model_config.data.image_size,
                                                  self.model_config.data.image_size)) * self.model_config.model.null_token
        return sampling_null_condition

    def _generate_condition(self, batch_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.scaler(batch_dict['precip_gt'])  # .to(config.device))

        if self.model_config.data.condition_mode == 0:
            raise AttributeError()  # deprecated
        elif self.model_config.data.condition_mode == 1:
            condition = batch_dict['precip_up']  #.to(config.device)
        elif self.model_config.data.condition_mode in [2, 6]:
            exclude_keys = ['precip_lr', 'precip_gt']
            tensors_to_stack = [tensor for key, tensor in batch_dict.items() if key not in exclude_keys]
            stacked_tensor = torch.cat(tensors_to_stack, dim=1)
            condition = stacked_tensor  #.to(config.device)
        elif self.model_config.data.condition_mode == 3:
            exclude_keys = ['precip_lr', 'precip_gt', 'precip_up']
            tensors_to_stack = [tensor for key, tensor in batch_dict.items() if key not in exclude_keys]
            stacked_tensor = torch.cat(tensors_to_stack, dim=1)
            condition = stacked_tensor  #.to(config.device)
        elif self.model_config.data.condition_mode == 4:
            condition = batch_dict['precip_masked']  #.to(config.device)
        elif self.model_config.data.condition_mode == 5:
            exclude_keys = ['precip_gt', 'mask']  # TODO check if including mask is useful
            tensors_to_stack = [tensor for key, tensor in batch_dict.items() if key not in exclude_keys]
            stacked_tensor = torch.cat(tensors_to_stack, dim=1)
            condition = stacked_tensor  #.to(config.device)
        else:
            raise AttributeError()
        return condition, y

    def _dropout_condition(self, condition: torch.Tensor) -> torch.Tensor:
        # implement dropout
        context_mask = torch.bernoulli(torch.zeros(condition.shape[0]) + (1 - self.model_config.model.drop_prob))
        context_mask = context_mask[:, None, None, None]  # shape: (batch_size, 1, 1, 1)
        context_mask = context_mask.to(self.device)  # shape: (batch_size,)
        condition = condition * context_mask  # shape: (batch_size, c, h, w)
        null_token_mask = torch.zeros_like(context_mask, device=self.device)
        null_token_mask[context_mask == 0] = self.model_config.model.null_token  # set to null token
        condition = condition + null_token_mask
        return condition


if __name__ == "__main__":
    _ = WassDiffLitModule(None, None, None, None)
