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
from src.models.ncsn import utils as mutils


class WassDiffLitModule(LightningModule):
    """
    """

    def __init__(self,
                 # optimizer: torch.optim.Optimizer,
                 # scheduler: torch.optim.lr_scheduler,
                 # compile: bool,
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

        # internal flags
        self.first_batch_visualized = False
        return

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        # Setup SDEs
        if self.model_config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=self.model_config.beta_min, beta_max=self.model_config.beta_max,
                                N=self.model_config.num_scales)
            sampling_eps = 1e-3
        elif self.model_config.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=self.model_config.beta_min, beta_max=self.model_config.beta_max,
                                   N=self.model_config.num_scales)
            sampling_eps = 1e-3
        elif self.model_config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=self.model_config.sigma_min, sigma_max=self.model_config.sigma_max,
                                N=self.model_config.num_scales)
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {self.hparams.sde} unknown.")

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

        optimizer = losses.get_optimizer(self.optimizer_config, self.net.parameters())

        # optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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

    def on_fit_start(self) -> None:
        """Lightning hook that is called when the fit begins."""
        ema = ExponentialMovingAverage(self.net.parameters(), decay=self.model_config.ema_rate)
        state = dict(optimizer=self.optimizer, model=self.net, ema=ema, step=0)
        # TODO: define scalar and inverse scalar

    def on_train_batch_start(self, batch, batch_idx) -> None:
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
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        raise NotImplementedError()

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        raise NotImplementedError()

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        raise NotImplementedError()

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


if __name__ == "__main__":
    _ = WassDiffLitModule(None, None, None, None)
