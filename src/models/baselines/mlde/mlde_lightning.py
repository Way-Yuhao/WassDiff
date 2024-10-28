from typing import Any, Dict, Tuple, Optional
# import wandb
import torch
from lightning import LightningModule
# from src.models.ncsn import ncsnpp_cond
from src.models.baselines.mlde import cncsnpp
# from src.models.ncsn.ema import ExponentialMovingAverage
from src.models.baselines.mlde.ema import ExponentialMovingAverage
# from src.utils.ncsn_utils import sde_lib
from src.utils.mlde_utils import sde_lib
# from src.utils.ncsn_utils import losses
from src.utils.mlde_utils import losses
# import src.utils.ncsn_utils.sampling as sampling
from src.utils.mlde_utils import sampling
# from src.models.ncsn import utils as mutils
from src.models.baselines.mlde import utils as mutils
from src.utils.ncsn_utils import datasets as datasets  # using NCSN
# sampling
# from src.utils.ncsn_utils.sampling import ReverseDiffusionPredictor, LangevinCorrector
# import src.utils.ncsn_utils.controllable_generation as controllable_generation
# from src.utils.ncsn_utils.utils import restore_checkpoint
# from src.utils.ncsn_utils.losses import get_optimizer
from src.utils.helper import yprint
# mlde
from src.models.baselines.mlde.location_params import LocationParams

class MLDELitModule(LightningModule):
    """
    """

    def __init__(self, model_config: dict, optimizer_config: dict,
                 compile: bool, num_samples: int = 1, pytorch_ckpt_path: Optional[str] = None,
                 *args, **kwargs) -> None:
        """Initialize a `MLDELitModule`.

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

        # attributes to be defined elsewhere
        self.train_step_fn = None
        self.eval_step_fn = None
        self.sampling_fn = None  # sampling function used during training (for displaying conditional samples)
        self.pc_upsampler = None  # sampling function used during inference (Predictor-Corrector upsampler)

        # internal flags
        self.automatic_optimization = False
        self.first_batch_visualized = False
        self.skip_next_batch = False  # flag to be modified by callbacks

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

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        self.model_config.device = self.device

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
                                                likelihood_weighting=likelihood_weighting)

        self.eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                               reduce_mean=reduce_mean, continuous=continuous,
                                               likelihood_weighting=likelihood_weighting)

        # Building sampling functions
        sampling_shape = (self.model_config.sampling.sampling_batch_size, self.model_config.data.num_channels,
                          self.model_config.data.image_size, self.model_config.data.image_size)
        self.sampling_fn = sampling.get_sampling_fn(self.model_config, sde, sampling_shape, sampling_eps)
        # s = self.model_config.sampling.sampling_batch_size
        # num_train_steps = self.model_config.training.n_iters

        return optimizer

    def configure_sampler(self) -> None:
        """
        Configure sampler at inference time.
        """
        # TODO: verify
        if self.hparams.pytorch_ckpt_path is not None:
            score_model = self.net
            optimizer = get_optimizer(self.optimizer_config, score_model.parameters())
            ema = ExponentialMovingAverage(score_model.parameters(),
                                           decay=self.model_config.model.ema_rate)
            state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)
            state = restore_checkpoint(self.hparams.pytorch_ckpt_path, state, self.device)
            ema.copy_to(score_model.parameters())
            yprint(f"\nRestored model from Pytorch ckpt: {self.hparams.pytorch_ckpt_path}")

        # sigmas = mutils.get_sigmas(self.model_config) # not used here or NCSN codebase
        self.scaler = datasets.get_data_scaler(self.model_config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.model_config)
        sde = 'VESDE'  # @param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
        if sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=self.model_config.model.sigma_min,
                                sigma_max=self.model_config.model.sigma_max,
                                N=self.model_config.model.num_scales)
        else:
            raise NotImplementedError()

        # FIXME: is this used at all?
        resolution_ratio = int(self.model_config.data.image_size / self.model_config.data.condition_size)
        predictor = ReverseDiffusionPredictor  # @param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
        corrector = LangevinCorrector  # @param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
        self.pc_upsampler = controllable_generation.get_pc_cfg_upsampler(sde,
                                                                         predictor, corrector,
                                                                         self.inverse_scaler,
                                                                         snr=self.model_config.sampling.snr,
                                                                         n_steps=self.model_config.sampling.n_steps_each,
                                                                         probability_flow=self.model_config.sampling.probability_flow,
                                                                         continuous=self.model_config.training.continuous,
                                                                         denoise=True)

    def on_fit_start(self) -> None:
        """Lightning hook that is called when the fit begins."""

        self.sampling_null_condition = self._generate_null_condition()
        self.state['optimizer'] = self.optimizers()  # TODO: verify

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of images.
        """
        return self.net(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        batch_dict, _ = batch  # discard coordinates
        condition, gt = self._generate_condition(batch_dict)
        condition, context_mask = self._dropout_condition(condition)
        loss = self.train_step_fn(self.state, gt, condition)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False, batch_size=condition.shape[0])
        step_output = {"batch_dict": batch_dict, 'condition': condition,
                       'context_mask': context_mask}
        return step_output

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch_dict, _ = batch  # discard coordinates
        condition, gt = self._generate_condition(batch_dict)
        eval_loss = self.eval_step_fn(self.state, gt, condition)
        self.log("val/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=False,
                 batch_size=condition.shape[0], sync_dist=True)
        step_output = {"batch_dict": batch_dict, "condition": condition}
        return step_output

    def on_test_start(self):
        self.configure_sampler()
        if self.hparams.compile:
            self.net = torch.compile(self.net)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        if self.skip_next_batch:  # determine whether to skip this batch
            return {}

        batch_dict, batch_coords, xr_low_res_batch, valid_mask = batch  # discard coordinates FIXME
        condition, gt = self._generate_condition(batch_dict)

        # ensemble prediction, if needed
        if self.hparams.num_samples > 1 and condition.shape[0] > 1:
            raise AttributeError('Ensemble prediction not supported for batch size > 1.')
        elif self.hparams.num_samples > 1 and condition.shape[0] == 1:
            condition = condition.repeat(self.hparams.num_samples, 1, 1, 1)

        null_condition = torch.ones_like(condition) * self.model_config.model.null_token
        batch_size = condition.shape[0]

        if self.hparams.bypass_sampling:
            batch_dict['precip_output'] = torch.zeros_like(gt)
        else:
            x = self.pc_upsampler(self.net, self.scaler(condition), w=self.model_config.model.w_guide,
                                  out_dim=(batch_size, 1, self.model_config.data.image_size, self.model_config.data.image_size),
                                  save_dir=None, null_condition=null_condition, gt=gt,
                                  display_pbar=self.hparams.display_sampling_pbar)
            if self.hparams.num_samples == 1:
                batch_dict['precip_output'] = x
            else:
                for i in range(self.hparams.num_samples):
                    batch_dict['precip_output_' + str(i)] = x[i, :, :, :]

        return {'batch_dict': batch_dict, 'batch_coords': batch_coords, 'xr_low_res_batch': xr_low_res_batch,
                'valid_mask': valid_mask}


    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Runs sampling fn when given condition tensor. No further trimming or processing is performance.
        Note that sampling fn is NOT the full Predictor-Corrector Sampler (more accurate).
        This is only intended to use during validation to gather quick samples.
        Usage at test time is not recommended.
        """
        # config = self.model_config
        # sampling_null_condition = self.sampling_null_condition
        sample, n = self.sampling_fn(self.net, condition)
        return sample
    def _generate_null_condition(self):
        # generate null condition for sampling
        s = self.model_config.sampling.sampling_batch_size
        if self.model_config.data.condition_mode in [1, 4]:
            sampling_null_condition = torch.ones(
                (s, self.model_config.data.num_channels, self.model_config.data.condition_size,
                 self.model_config.data.condition_size)) * self.model_config.model.null_token
        elif self.model_config.data.condition_mode in [2, 5, 6]:
            sampling_null_condition = torch.ones(
                (s, self.model_config.data.num_channels + self.model_config.data.num_context_chs,
                 self.model_config.data.image_size,
                 self.model_config.data.image_size)) * self.model_config.model.null_token
        elif self.model_config.data.condition_mode == 3:
            sampling_null_condition = torch.ones(
                (s, self.model_config.data.num_context_chs, self.model_config.data.image_size,
                 self.model_config.data.image_size)) * self.model_config.model.null_token
        return sampling_null_condition.to(self.device)

    def _generate_condition(self, batch_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.scaler(batch_dict['precip_gt'])  # .to(config.device))

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

    def _dropout_condition(self, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # implement dropout
        context_mask = torch.bernoulli(torch.zeros(condition.shape[0]) + (1 - self.model_config.model.drop_prob))
        context_mask = context_mask[:, None, None, None]  # shape: (batch_size, 1, 1, 1)
        context_mask = context_mask.to(self.device)  # shape: (batch_size,)
        condition = condition * context_mask  # shape: (batch_size, c, h, w)
        null_token_mask = torch.zeros_like(context_mask, device=self.device)
        null_token_mask[context_mask == 0] = self.model_config.model.null_token  # set to null token
        condition = condition + null_token_mask
        return condition, context_mask


if __name__ == "__main__":
    _ = WassDiffLitModule(None, None, None, None)
