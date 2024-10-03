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

# from src.utils.corrector_gan_utils.utils import tqdm, device
from src.models.baselines.corrector_gan.layers import *
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
                 model_config: Dict = {}, automatic_optimization: bool = False):  # fill in
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

        if disc_spectral_norm:
            self.disc.apply(self.add_sn)
        if gen_spectral_norm:
            self.gen.apply(self.add_sn)

        # additional params absent in the original code
        self.automatic_optimization = automatic_optimization
        self.model_config = model_config

        # to be defined elsewhere
        self.rainfall_dataset = None

        self.save_hyperparameters()

    def configure_optimizers(self):

        ### additional steps related to WassDiff dataset
        # Create data normalizer and its inverse
        self.scaler = datasets.get_data_scaler(self.model_config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.model_config)
        #######

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
        return [disc_opt, gen_opt]

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
            opt_d.step()
            self.log('discriminator_loss', loss_disc, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        # Update generator `gen_freq` times
        for _ in range(self.opt_hparams['gen_freq']):
            opt_g.zero_grad()
            noise = torch.randn(real.shape[0], *self.noise_shape[-3:], device=self.device)
            fake, corrected_lr = self.gen(condition, noise)
            disc_fake = self.disc(condition, fake).reshape(-1)
            loss_gen = self.compute_generator_loss(disc_fake, fake, real, corrected_lr)
            self.manual_backward(loss_gen)
            opt_g.step()
            self.log('generator_loss', loss_gen, on_epoch=True, on_step=False, prog_bar=True, logger=True)


    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch_dict, _ = batch
        x, y = self._generate_condition(batch_dict)
        x = self._downscale_condition(x)

        for i in range(self.val_hparams['val_nens']):
            noise = torch.randn(y.shape[0], * self.noise_shape[-3:], device=self.device)
            pred, lr_corrected_pred = self.gen(x, noise)  # TODO: debug
            pred = pred.detach().to('cpu').numpy().squeeze()
            batch_dict['precip_output_' + str(i)] = torch.from_numpy(pred).unsqueeze(1)

        step_output = {"batch_dict": batch_dict, "condition": x}
        # TODO: calculate and report validation metrics
        return step_output

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
            loss_gen += self.loss_hparams['gen_loss']["ens_mean_lr_corrected_l1"] * gen_ens_mean_lr_corrected_l1(
                corrected_lr, real)

        return loss_gen

    def _downscale_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """
        WassDiff dataloder returns upsampled condition that matches the dim out expected output. This function
        downscales the condition to the reduced size
        """
        return F.interpolate(condition, size=(self.noise_shape[-2], self.noise_shape[-1]), mode='bilinear')
