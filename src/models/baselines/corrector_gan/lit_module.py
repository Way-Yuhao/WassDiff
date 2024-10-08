from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from lightning import LightningModule
import torch.optim as optim
# import torchvision
import matplotlib.pyplot as plt
# import xarray as xr
# import xskillscore as xs

from src.models.baselines.corrector_gan.layers import *
from src.models.baselines.corrector_gan.models import Corrector
# from src.utils.corrector_gan_utils.dataloader import log_retrans
from src.utils.corrector_gan_utils.loss import *
from src.utils.ncsn_utils import datasets as datasets
from src.utils.metrics import calc_mae, calc_bias, calc_crps


class CorrectorGan(LightningModule):
    def __init__(self, generator, discriminator, noise_shape, input_channels=1,
                 cond_idx=0, real_idx=1,
                 disc_spectral_norm=False, gen_spectral_norm=False, zero_noise=False,
                 opt_hparams={'gen_optimiser': 'adam', 'disc_optimiser': 'adam', 'disc_lr': 1e-4, 'gen_lr': 1e-4,
                              'gen_freq': 1, 'disc_freq': 5, 'b1': 0.0, 'b2': 0.9},
                 loss_hparams={'disc_loss': "wasserstein", 'gen_loss': "wasserstein", 'lambda_gp': 10},
                 val_hparams={'val_nens': 10, 'tp_log': 0.01, 'ds_max': 50, 'ds_min': 0},
                 model_config: Dict = {}, automatic_optimization: bool = False,
                 use_gradient_clipping: bool = False, gen_ckpt: Optional[str] = None, *args, **kwargs):  # fill in
        super().__init__()

        self.noise_shape = noise_shape
        self.gen = generator(input_channels=input_channels)
        self.disc = discriminator(input_channels=input_channels)
        self.real_idx = real_idx
        self.cond_idx = cond_idx
        self.opt_hparams = opt_hparams
        self.loss_hparams = loss_hparams
        self.input_channels = input_channels
        self.val_hparams = val_hparams
        self.upsample_input = nn.Upsample(scale_factor=8)
        self.zero_noise = zero_noise
        self.use_gradient_clipping = use_gradient_clipping
        self.gen_ckpt = gen_ckpt

        if disc_spectral_norm:
            self.disc.apply(self.add_sn)
        if gen_spectral_norm:
            self.gen.apply(self.add_sn)

        # additional params absent in the original code
        self.automatic_optimization = automatic_optimization
        self.model_config = model_config

        # to be defined elsewhere
        self.rainfall_dataset = None
        self.skip_next_batch = False  # flag to be modified by callbacks
        self.scaler = None
        self.inverse_scaler = None

        self.save_hyperparameters(logger=False, ignore=("model_config", "optimizer_config"))

    def setup(self, stage: str) -> None:
        # resume from stage 2 (GAN high-res pre-training)
        if self.gen_ckpt is not None:
            print("Loading Generator weights...")
            checkpoint = torch.load(self.gen_ckpt)
            loaded_state_dict = checkpoint['state_dict']
            new_state_dict = {}
            prefix = 'gen.'
            for key, value in loaded_state_dict.items():
                # remove the prefix
                new_key = key[len(prefix):]
                new_state_dict[new_key] = value
            self.gen.load_state_dict(new_state_dict)

        ### additional steps related to WassDiff dataset
        # Create data normalizer and its inverse
        self.scaler = datasets.get_data_scaler(self.model_config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.model_config)
        #######
    def configure_optimizers(self):

        if self.opt_hparams['gen_optimiser'] == 'adam':
            gen_opt = optim.Adam(self.gen.parameters(), eps=5e-5, lr=self.opt_hparams['gen_lr'],
                                 betas=(self.opt_hparams['b1'], self.opt_hparams['b2']), weight_decay=1e-4)

        elif self.opt_hparams['gen_optimiser'] == 'sgd':
            gen_opt = optim.SGD(self.gen.parameters(), lr=self.opt_hparams['gen_lr'],
                                momentum=self.opt_hparams['gen_momentum'])
        else:
            raise NotImplementedError
        if self.opt_hparams['disc_optimiser'] == 'adam':
            disc_opt = optim.Adam(self.disc.parameters(), eps=5e-5, lr=self.opt_hparams['disc_lr'],
                                  betas=(self.opt_hparams['b1'], self.opt_hparams['b2']))
        elif self.opt_hparams['disc_optimiser'] == 'sgd':
            disc_opt = optim.SGD(self.disc.parameters(), lr=self.opt_hparams['disc_lr'],
                                 momentum=self.opt_hparams['disc_momentum'])
        else:
            raise NotImplementedError
        # return [{"optimizer": disc_opt, "frequency": self.opt_hparams['disc_freq']},
        #         {"optimizer": gen_opt, "frequency": self.opt_hparams['gen_freq']}]
        return [gen_opt, disc_opt]

    def on_validation_start(self) -> None:
        self.rainfall_dataset = self.trainer.datamodule.precip_dataset

    def add_sn(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(m)
        else:
            m

    def forward(self, condition, noise):
        return self.gen(condition, noise)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        opt_g, opt_d = self.optimizers()
        # condition, real = batch[self.cond_idx], batch[self.real_idx]
        batch_dict, _ = batch
        condition, real = self._generate_condition(batch_dict)
        condition = self._downscale_condition(condition)

        # Update discriminator `disc_freq` times
        for _ in range(self.opt_hparams['disc_freq']):
            opt_d.zero_grad()
            noise = torch.randn(real.shape[0], *self.noise_shape[-3:], device=self.device)
            fake, _ = self.gen(condition, noise)
            fake = fake.detach()
            disc_real = self.disc(condition, real).reshape(-1)
            disc_fake = self.disc(condition, fake.detach()).reshape(-1)
            loss_disc = self.compute_discriminator_loss(disc_real, disc_fake, condition, real, fake)
            self.manual_backward(loss_disc)

            if self.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)

            opt_d.step()
            self.log('train/D_loss', loss_disc, batch_size=real.shape[0],
                     on_epoch=True, on_step=False, prog_bar=True, logger=True)
            self.log('train/D_real', torch.mean(disc_real), batch_size=real.shape[0],
                        on_epoch=True, on_step=False, prog_bar=True, logger=True)
            self.log('train/D_fake', torch.mean(disc_fake), batch_size=real.shape[0],
                        on_epoch=True, on_step=False, prog_bar=True, logger=True)

        # Update generator `gen_freq` times
        for _ in range(self.opt_hparams['gen_freq']):
            opt_g.zero_grad()
            noise = torch.randn(real.shape[0], *self.noise_shape[-3:], device=self.device)
            fake, corrected_lr = self.gen(condition, noise)
            disc_fake = self.disc(condition, fake).reshape(-1)
            loss_gen = self.compute_generator_loss(disc_fake, fake, real, corrected_lr)
            self.manual_backward(loss_gen)

            if self.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)

            opt_g.step()
            self.log('train/G_loss', loss_gen, batch_size=real.shape[0],
                     on_epoch=True, on_step=False, prog_bar=True, logger=True)


    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch_dict, _ = batch
        x, y = self._generate_condition(batch_dict)
        x = self._downscale_condition(x)

        for i in range(self.val_hparams['val_nens']):
            noise = torch.randn(y.shape[0], * self.noise_shape[-3:], device=self.device)
            pred, lr_corrected_pred = self.gen(x, noise)
            pred = pred.detach().to('cpu').numpy().squeeze()
            batch_dict['precip_output_' + str(i)] = torch.from_numpy(pred).unsqueeze(1)

        step_output = {"batch_dict": batch_dict, "condition": x}
        # TODO: calculate and report validation metrics
        return step_output

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        if self.skip_next_batch:  # determine whether to skip this batch
            return {}
        batch_dict, batch_coords, xr_low_res_batch, valid_mask = batch  # discard coordinates FIXME
        condition, gt = self._generate_condition(batch_dict)
        condition = self._downscale_condition(condition)
        # batch_size = condition.shape[0]

        # ensemble prediction, if needed
        if self.hparams.num_samples > 1 and condition.shape[0] > 1:
            raise AttributeError('Ensemble prediction not supported for batch size > 1.')
        elif self.hparams.num_samples > 1 and condition.shape[0] == 1:
            condition = condition.repeat(self.hparams.num_samples, 1, 1, 1)

        # if self.hparams.bypass_sampling:
        #     batch_dict['precip_output'] = torch.zeros_like(gt)
        # else:
        noise = torch.randn(gt.shape[0], *self.noise_shape[-3:], device=self.device)
        pred, lr_corrected_pred = self.gen(condition, noise)
        if self.hparams.num_samples == 1:
            batch_dict['precip_output'] = pred
        else:
            for i in range(self.hparams.num_samples):
                batch_dict['precip_output_' + str(i)] = pred[i, :, :, :]
        return {'batch_dict': batch_dict, 'batch_coords': batch_coords, 'xr_low_res_batch': xr_low_res_batch,
                'valid_mask': valid_mask}

    def on_validation_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        batch_dict = self.rainfall_dataset.inverse_normalize_batch(outputs['batch_dict'])
        batch_size = batch_dict['precip_gt'].shape[0]

        for i in range(batch_size):
            precip_output, precip_gt = None, None
            for key, item in outputs['batch_dict'].items():
                if 'precip_output' in key:
                    cur_precip = item[i, 0, :, :].cpu().detach().numpy()
                    cur_precip = np.expand_dims(cur_precip, axis=0)
                    if precip_output is None:
                        precip_output = cur_precip
                    else:
                        precip_output = np.concatenate([precip_output, cur_precip], axis=0)
                elif key == 'precip_gt':
                    if precip_gt is None:
                        precip_gt = item[i, 0, :, :].cpu().detach().numpy()
                    else:
                        # check if the gt is the same for all runs
                        assert np.allclose(precip_gt, item[i, 0, :, :].cpu().detach().numpy())
            precip_output_avg = np.mean(precip_output, axis=0)
            mae = calc_mae(precip_output_avg, precip_gt, valid_mask=None, k=1, pooling_func='mean')
            bias = calc_bias(precip_output_avg, precip_gt, valid_mask=None, k=1, pooling_func='mean')
            crps = calc_crps(precip_output, precip_gt)
            self.log('val/mae', mae, on_epoch=True, on_step=False, prog_bar=False, logger=True)
            self.log('val/bias', bias, on_epoch=True, on_step=False, prog_bar=False, logger=True)
            self.log('val/crps', crps, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return

    def sample(self, condition: torch.Tensor):
        noise = torch.randn(condition.shape[0], *self.noise_shape[-3:], device=self.device)
        fake, _ = self.gen(condition, noise)
        return fake


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

    def compute_discriminator_loss(self, disc_real, disc_fake, condition, real, fake):
        loss_disc = torch.tensor(0.0).to(self.device)

        if "wasserstein" in self.loss_hparams['disc_loss']:
            loss_disc += self.loss_hparams['disc_loss']["wasserstein"] * disc_wasserstein(disc_real, disc_fake)

        if "hinge" in self.loss_hparams['disc_loss']:
            loss_disc += self.loss_hparams['disc_loss']["hinge"] * disc_hinge(disc_real, disc_fake)

        if "gradient_penalty" in self.loss_hparams['disc_loss']:
            loss_disc += self.loss_hparams['disc_loss']["gradient_penalty"] * gradient_penalty(
                self.disc, condition, real, fake, self.device)

        return loss_disc

    def compute_generator_loss(self, disc_fake, fake, real, corrected_lr):
        loss_gen = torch.tensor(0.0).to(self.device)
        if "wasserstein" in self.loss_hparams['gen_loss']:
            loss_gen += self.loss_hparams['gen_loss']["wasserstein"] * gen_wasserstein(disc_fake)
        if "non_saturating" in self.loss_hparams['gen_loss']:
            loss_gen += self.loss_hparams['gen_loss']["non_saturating"] * gen_logistic_nonsaturating(disc_fake)
        if "ens_mean_L1_weighted" in self.loss_hparams['gen_loss']:
            loss_gen += self.loss_hparams['gen_loss']["ens_mean_L1_weighted"] * gen_ens_mean_L1_weighted(fake, real)
        if "lr_corrected_skill" in self.loss_hparams['gen_loss']:
            loss_gen += self.loss_hparams['gen_loss']["lr_corrected_skill"] * gen_lr_corrected_skill(corrected_lr, real)
        if "lr_corrected_l1" in self.loss_hparams['gen_loss']:
            loss_gen += self.loss_hparams['gen_loss']["lr_corrected_l1"] * gen_lr_corrected_l1(corrected_lr, real)
        if "ens_mean_lr_corrected_l1" in self.loss_hparams['gen_loss']:
            # This line below throws a broadcasting warning; believed to be irrelevant -- Yuhao
            loss_gen += self.loss_hparams['gen_loss']["ens_mean_lr_corrected_l1"] * gen_ens_mean_lr_corrected_l1(
                corrected_lr, real)
        return loss_gen

    def _downscale_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """
        WassDiff dataloader returns upsampled condition that matches the dim out expected output. This function
        downscales the condition to the reduced size
        """
        return F.interpolate(condition, size=(self.noise_shape[-2], self.noise_shape[-1]), mode='bilinear')


class GANGenerator(LightningModule):

    def __init__(self, generator, noise_shape, input_channels=1, cond_idx=0, real_idx=1,
                 gen_spectral_norm=False, zero_noise=False,
                 # opt_hparams={'gen_optimiser': 'adam', 'disc_optimiser': 'adam', 'disc_lr': 1e-4, 'gen_lr': 1e-4,
                 #              'gen_freq': 1, 'disc_freq': 5, 'b1': 0.0, 'b2': 0.9},
                 # loss_hparams={'disc_loss': "wasserstein", 'gen_loss': "wasserstein", 'lambda_gp': 10},
                 # val_hparams={'val_nens': 10, 'tp_log': 0.01, 'ds_max': 50, 'ds_min': 0},
                 model_config: Dict = {}, automatic_optimization: bool = False,
                 use_gradient_clipping: bool = False, corrector_ckpt: Optional[str] = None):  # fill in
        super().__init__()

        self.noise_shape = noise_shape
        self.gen = generator(input_channels=input_channels)
        self.real_idx = real_idx
        self.cond_idx = cond_idx
        # self.opt_hparams = opt_hparams
        # self.loss_hparams = loss_hparams
        self.input_channels = input_channels
        # self.val_hparams = val_hparams
        self.upsample_input = nn.Upsample(scale_factor=8)
        self.zero_noise = zero_noise
        self.use_gradient_clipping = use_gradient_clipping
        self.corrector_ckpt = corrector_ckpt

        if gen_spectral_norm:
            self.gen.apply(self.add_sn)

        # additional params absent in the original code
        self.automatic_optimization = automatic_optimization
        self.model_config = model_config
        self.l1loss = nn.L1Loss()

        # to be defined elsewhere
        self.rainfall_dataset = None
        self.scaler = None
        self.inverse_scaler = None
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        print("Initializing G weights...")
        self.gen.initialize_weights()
        if self.corrector_ckpt is not None:
            print("Loading Corrector weights...")
            checkpoint = torch.load(self.corrector_ckpt)
            # self.gen.corrector.load_state_dict(weight['state_dict'])
            loaded_state_dict = checkpoint['state_dict']
            # Initialize a new state_dict for the Conv2d layer
            conv_state_dict = {}
            prefix = 'corrector.final.'
            # Extract and map the relevant keys
            for key in loaded_state_dict:
                if key == f'{prefix}weight':
                    conv_state_dict['weight'] = loaded_state_dict[key]
                elif key == f'{prefix}bias':
                    conv_state_dict['bias'] = loaded_state_dict[key]

            # Check if both weight and bias are extracted
            if 'weight' in conv_state_dict and 'bias' in conv_state_dict:
                # Load the state_dict into the Conv2d layer
                self.gen.corrector.load_state_dict(conv_state_dict)
            else:
                raise KeyError("Required keys 'weight' and 'bias' not found in the loaded state_dict.")

    def configure_optimizers(self):
        ### additional steps related to WassDiff dataset
        # Create data normalizer and its inverse
        self.scaler = datasets.get_data_scaler(self.model_config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.model_config)
        #######
        opt = optim.Adam(self.gen.parameters(), eps=5e-5, lr=5e-5, betas=(0, 0.9), weight_decay=1e-4)
        return opt

    def loss(self, real, real_lr, corrected, fake):
        gen_l1_loss = self.l1loss(real, fake)
        corrector_l1_loss = self.l1loss(real_lr, corrected)
        loss = gen_l1_loss + corrector_l1_loss
        loss_dict = {'gen_l1_loss': gen_l1_loss, 'corrector_l1_loss': corrector_l1_loss}
        return loss, loss_dict


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch_dict, _ = batch
        condition, real = self._generate_condition(batch_dict)
        condition = self._downscale_condition(condition)
        real_lr = self._downscale_condition(real)

        if self.zero_noise:
            noise = torch.zeros(real.shape[0], *self.noise_shape[-3:], device=self.device)
        else:
            noise = torch.randn(real.shape[0], *self.noise_shape[-3:], device=self.device)
        fake, corrected_lr = self.gen(condition, noise)
        loss, loss_dict = self.loss(real, real_lr, corrected_lr, fake)

        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True, batch_size=real.shape[0])
        self.log('train/gen_l1_loss', loss_dict['gen_l1_loss'], on_epoch=True, on_step=False, prog_bar=True, logger=True, batch_size=real.shape[0])
        self.log('train/corrector_l1_loss', loss_dict['corrector_l1_loss'], on_epoch=True, on_step=False, prog_bar=True, logger=True, batch_size=real.shape[0])
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch_dict, _ = batch
        condition, real = self._generate_condition(batch_dict)
        condition = self._downscale_condition(condition)
        real_lr = self._downscale_condition(real)

        if self.zero_noise:
            noise = torch.zeros(real.shape[0], *self.noise_shape[-3:], device=self.device)
        else:
            noise = torch.randn(real.shape[0], *self.noise_shape[-3:], device=self.device)
        fake, corrected_lr = self.gen(condition, noise)
        loss, loss_dict = self.loss(real, real_lr, corrected_lr, fake)

        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True, batch_size=real.shape[0])
        self.log('val/gen_l1_loss', loss_dict['gen_l1_loss'], on_epoch=True, on_step=False, prog_bar=True, logger=True, batch_size=real.shape[0])
        self.log('val/corrector_l1_loss', loss_dict['corrector_l1_loss'], on_epoch=True, on_step=False, prog_bar=True, logger=True, batch_size=real.shape[0])

        step_output = {"batch_dict": batch_dict, "condition": condition}
        return step_output

    def sample(self, condition: torch.Tensor):
        noise = torch.randn(condition.shape[0], *self.noise_shape[-3:], device=self.device)
        fake, _ = self.gen(condition, noise)
        return fake

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

    def _downscale_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """
        WassDiff dataloader returns upsampled condition that matches the dim out expected output. This function
        downscales the condition to the reduced size
        """
        return F.interpolate(condition, size=(self.noise_shape[-2], self.noise_shape[-1]), mode='bilinear')


class CheckCorrector(LightningModule):
    """Stage 1 pretrained model for the CorrectorGAN model"""
    def __init__(self, input_channels=15, cond_idx=0, real_idx=1, model_config: Dict = {}):
        super().__init__()
        self.cond_idx = cond_idx
        self.real_idx = real_idx
        self.corrector = Corrector(input_channels=input_channels)
        self.downsample = F.interpolate
        #         self.loss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.l1_lambda = 0.0000
        self.fss_window = 4
        self.fss_lambda = 1
        self.fss_pool = nn.AvgPool2d(kernel_size=self.fss_window, padding=0, stride=1)
        # self.fss_pool.register_full_backward_hook(self.printgradnorm)
        self.sigmoid = nn.Sigmoid()
        # self.fss_pool.register_full_backward_hook(self.printgradnorm)
        # self.fss_dbg = DBGModule()
        # self.fss_dbg.register_full_backward_hook(self.printgradnorm)

        # additional params absent in the original code
        self.model_config = model_config
        self.scaler = None
        self.inverse_scaler = None

    def configure_optimizers(self):
        ### additional steps related to WassDiff dataset
        # Create data normalizer and its inverse
        self.scaler = datasets.get_data_scaler(self.model_config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.model_config)
        #######
        opt = optim.Adam(self.corrector.parameters(), eps=5e-5, lr=5e-5, betas=(0, 0.9), weight_decay=1e-4)
        return opt

    def loss(self, real, corrected):
        #        if torch.sum(torch.isnan(self.fss(corrected, real, threshold = 0.5)))>0:
        #            print("nan fss")
        #        if torch.sum(torch.isnan(self.l1loss(real, corrected)))>0:
        #            print("nan l1loss")
        return self.l1loss(real, corrected) + self.fss_lambda * self.fss(corrected, real, threshold=0.5)

    def forward(self, condition, noise):
        return self.gen(condition, noise)

    def training_step(self, batch, batch_idx):
        batch_dict, _ = batch
        condition, real = self._generate_condition(batch_dict)
        # condition = self._downscale_condition(condition)

        #        if torch.sum(torch.isnan(condition))>0:
        #            print("nan condition")

        #        if torch.sum(torch.isnan(real))>0:
        #            print("nan real")

        real = self.downsample(real, scale_factor=0.125, mode='bilinear', align_corners=False)
        condition = self.downsample(condition, scale_factor=0.125, mode='bilinear',
                                    align_corners=False)  # modified by Yuhao

        corrected = self.corrector(condition)

        loss = self.loss(real, corrected).mean()  # + self.l1_lambda*torch.linalg.norm(corrected.view(-1), 1)

        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True,
                 batch_size=real.shape[0])

        if self.global_step % 100 == 0:
            l1error = F.l1_loss(real, corrected)
            forecast_l1error = F.l1_loss(real, condition[:, 0:1, :, :])
            self.log('train/forecast_loss', forecast_l1error, on_epoch=True, on_step=False, prog_bar=False,
                     logger=True, batch_size=real.shape[0])
            self.log('train/our_error_minus_forecast_error', l1error - forecast_l1error, on_epoch=True,
                     on_step=False, prog_bar=False, logger=True, batch_size=real.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict, _ = batch
        condition, real = self._generate_condition(batch_dict)
        # condition = self._downscale_condition(condition)

        condition = self.downsample(condition, scale_factor=0.125, mode='bilinear', align_corners=False)  # modified by Yuhao

        real = self.downsample(real, scale_factor=0.125, mode='bilinear', align_corners=False)
        corrected = self.corrector(condition)

        l1error = F.l1_loss(real, corrected)
        forecast_l1error = F.l1_loss(real, condition[:, 0:1, :, :])

        self.log('val/loss', l1error, on_epoch=True, prog_bar=True, logger=True, batch_size=real.shape[0])
        self.log('val/forecast_loss', forecast_l1error, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=real.shape[0])
        self.log('val/our_error_minus_forecast_error', l1error - forecast_l1error, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=real.shape[0])

        forecast_fss_10 = self.fss(condition, real, threshold=0.1)
        forecast_fss_50 = self.fss(condition, real, threshold=0.5)
        corrected_fss_10 = self.fss(corrected, real, threshold=0.1)
        corrected_fss_50 = self.fss(corrected, real, threshold=0.5)

        self.log('val/forecast_fss_10', forecast_fss_10.mean(), on_epoch=True, prog_bar=False, logger=True,
                 batch_size=real.shape[0])
        self.log('val/forecast_fss_50', forecast_fss_50.mean(), on_epoch=True, prog_bar=False, logger=True,
                 batch_size=real.shape[0])
        self.log('val/corrected_fss_10', corrected_fss_10.mean(), on_epoch=True, prog_bar=False, logger=True,
                 batch_size=real.shape[0])
        self.log('val/corrected_fss_50', corrected_fss_50.mean(), on_epoch=True, prog_bar=False, logger=True,
                 batch_size=real.shape[0])
        step_output = {"batch_dict": batch_dict, "condition": condition}
        return step_output

    def print_grad_norm(self, module, grad_input, grad_output):
        print('Inside ' + module.__class__.__name__ + ' backward')
        print('Inside class:' + self.__class__.__name__)
        print('')
        print('grad_input: ', type(grad_input))
        print('grad_input[0]: ', torch.reshape(grad_input[0], (-1,)))
        print('grad_output: ', type(grad_output))
        print('grad_output[0]: ', grad_output[0])
        print('')
        print('grad_input size:', grad_input[0].size())
        print('grad_output size:', grad_output[0].size())
        print('grad_input norm:', grad_input[0].norm())
        print('grad_output norm:', grad_output[0].norm())
        print('')
        print('grad_input num nans:', torch.sum(torch.isnan(grad_input[0])))
        print('grad_output num nans:', torch.sum(torch.isnan(grad_output[0])))

    def fss(self, x, y, threshold):
        c = 200
        x_mask = self.sigmoid(c * (x - threshold))
        y_mask = self.sigmoid(c * (y - threshold))

        y_out = self.fss_pool(y_mask)
        x_out = self.fss_pool(x_mask[:, 0:1, :, :])

        mse_sample = torch.mean(torch.square(x_out - y_out), dim=[1, 2, 3])
        # mse_ref = torch.mean(torch.square(x_out), dim=[1,2,3]) +  torch.mean(torch.square(y_out), dim=[1,2,3])

        # mse_ref = self.fss_dbg(x_out, y_out)
        # nonzero_mseref = mse_ref!=0
        # fss = 1 - torch.divide(mse_sample[nonzero_mseref], mse_ref[nonzero_mseref])

        # return torch.mean(fss)
        return mse_sample


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

    def sample(self, condition: torch.Tensor):
        return self.corrector(condition)