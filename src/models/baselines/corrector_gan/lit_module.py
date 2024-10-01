import numpy as np
import torch
from torch import nn
from lightning import LightningModule
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import xarray as xr
import xskillscore as xs
from dask.diagnostics import ProgressBar
from sklearn.metrics import f1_score

from src.utils.corrector_gan_utils.utils import tqdm, device
from src.models.baselines.corrector_gan.layers import *
from src.utils.corrector_gan_utils.dataloader import log_retrans
from src.utils.corrector_gan_utils.loss import *


class CorrectorGan(LightningModule):
    def __init__(self, generator, discriminator, noise_shape, input_channels=1,
                 cond_idx=0, real_idx=1,
                 disc_spectral_norm=False, gen_spectral_norm=False, zero_noise=False,
                 opt_hparams={'gen_optimiser': 'adam', 'disc_optimiser': 'adam', 'disc_lr': 1e-4, 'gen_lr': 1e-4,
                              'gen_freq': 1, 'disc_freq': 5, 'b1': 0.0, 'b2': 0.9},
                 loss_hparams={'disc_loss': "wasserstein", 'gen_loss': "wasserstein", 'lambda_gp': 10},
                 val_hparams={'val_nens': 10, 'tp_log': 0.01, 'ds_max': 50, 'ds_min': 0}):  # fill in
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

        self.save_hyperparameters()

    def add_sn(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(m)
        else:
            m

    def forward(self, condition, noise):
        return self.gen(condition, noise)

    def training_step(self, batch, batch_idx, optimizer_idx):

        condition, real = batch[self.cond_idx], batch[self.real_idx]

        if self.global_step % 500 == 0:
            self.gen.eval()
            noise = torch.randn(real.shape[0], *self.noise_shape[-3:], device=self.device)
            #         # log sampled images
            sample_imgs, sample_corrected_lrs = self.gen(condition, noise)
            sample_imgs = torch.cat([real, sample_imgs], dim=0)
            #                 print(sample_imgs.shape)
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, self.global_step)
            sample_corrected_lrs = torch.cat(
                [self.upsample_input(F.interpolate(real, scale_factor=0.125, mode='bilinear', align_corners=False)),
                 self.upsample_input(condition[:, 0:1, :, :]), self.upsample_input(sample_corrected_lrs)], dim=0)
            grid = torchvision.utils.make_grid(sample_corrected_lrs)
            self.logger.experiment.add_image('corrected_lr_forecasts', grid, self.global_step)

            if self.input_channels > 1:
                input_forcasts = self.upsample_input(condition)
                #                     print(input_forcasts.view(-1, input_forcasts.shape[2], input_forcasts.shape[3]).unsqueeze(1).shape)
                grid = torchvision.utils.make_grid(
                    input_forcasts.view(-1, input_forcasts.shape[2], input_forcasts.shape[3]).unsqueeze(1),
                    nrow=self.input_channels)
            else:
                grid = torchvision.utils.make_grid(condition)
            self.logger.experiment.add_image('input_images', grid, self.global_step)
            self.gen.train()

        #         # train discriminator
        if optimizer_idx == 0:
            if self.zero_noise:
                noise = torch.zeros(real.shape[0], *self.noise_shape[-3:], device=self.device)
            else:
                noise = torch.randn(real.shape[0], *self.noise_shape[-3:], device=self.device)
            disc_real = self.disc(condition, real).reshape(-1)
            if len(noise.shape) == 5:
                fake = []
                disc_fake = []
                corrected_lr = []
                for i in range(noise.shape[1]):
                    noise_sample = noise[:, i, :, :, :]
                    fake_temp, corrected_lr_temp = self.gen(condition, noise_sample)
                    disc_fake_temp = self.disc(condition, fake_temp).reshape(-1)
                    fake.append(fake_temp)
                    disc_fake.append(disc_fake_temp)
                    corrected_lr.append(corrected_lr_temp)
                fake = torch.stack(fake, dim=0)
                disc_fake = torch.stack(disc_fake, dim=0)
                corrected_lr = torch.stack(corrected_lr, dim=0)
            else:
                fake, corrected_lr = self.gen(condition, noise)
                disc_fake = self.disc(condition, fake).reshape(-1)

            loss_disc = torch.tensor(0.0).to(self.device)

            if "wasserstein" in self.loss_hparams['disc_loss']:
                loss_disc += self.loss_hparams['disc_loss']["wasserstein"] * disc_wasserstein(disc_real, disc_fake)

            if "hinge" in self.loss_hparams['disc_loss']:
                loss_disc += self.loss_hparams['disc_loss']["hinge"] * disc_hinge(disc_real, disc_fake)

            if "gradient_penalty" in self.loss_hparams['disc_loss']:
                loss_disc += self.loss_hparams['disc_loss']["gradient_penalty"] * gradient_penalty(self.disc, condition,
                                                                                                   real, fake,
                                                                                                   self.device)

            self.log('discriminator_loss', loss_disc, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_disc

        #         #train generator
        elif optimizer_idx == 1:
            #             print(self.gen.training)
            if self.zero_noise:
                noise = torch.zeros(real.shape[0], *self.noise_shape, device=self.device)
            else:
                noise = torch.randn(real.shape[0], *self.noise_shape, device=self.device)
            if len(noise.shape) == 5:
                fake = []
                disc_fake = []
                corrected_lr = []
                for i in range(noise.shape[1]):
                    noise_sample = noise[:, i, :, :, :]
                    fake_temp, corrected_lr_temp = self.gen(condition, noise_sample)
                    disc_fake_temp = self.disc(condition, fake_temp).reshape(-1)
                    fake.append(fake_temp)
                    disc_fake.append(disc_fake_temp)
                    corrected_lr.append(corrected_lr_temp)
                fake = torch.stack(fake, dim=0)
                disc_fake = torch.stack(disc_fake, dim=0)
                corrected_lr = torch.stack(corrected_lr, dim=0)

            else:
                fake, corrected_lr = self.gen(condition, noise)
                disc_fake = self.disc(condition, fake).reshape(-1)

            loss_gen = torch.tensor(0.0).to(self.device)

            if "wasserstein" in self.loss_hparams['gen_loss']:
                loss_gen += self.loss_hparams['gen_loss']["wasserstein"] * gen_wasserstein(disc_fake)

            if "non_saturating" in self.loss_hparams['gen_loss']:
                loss_gen += self.loss_hparams['gen_loss']["non_saturating"] * gen_logistic_nonsaturating(disc_fake)

            if "ens_mean_L1_weighted" in self.loss_hparams['gen_loss']:
                loss_gen += self.loss_hparams['gen_loss']["ens_mean_L1_weighted"] * gen_ens_mean_L1_weighted(fake, real)

            if "lr_corrected_skill" in self.loss_hparams['gen_loss']:
                loss_gen += self.loss_hparams['gen_loss']["lr_corrected_skill"] * gen_lr_corrected_skill(corrected_lr,
                                                                                                         real)

            if "lr_corrected_l1" in self.loss_hparams['gen_loss']:
                loss_gen += self.loss_hparams['gen_loss']["lr_corrected_l1"] * gen_lr_corrected_l1(corrected_lr, real)

            if "ens_mean_lr_corrected_l1" in self.loss_hparams['gen_loss']:
                loss_gen += self.loss_hparams['gen_loss']["ens_mean_lr_corrected_l1"] * gen_ens_mean_lr_corrected_l1(
                    corrected_lr, real)

            self.log('generator_loss', loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)
            return loss_gen

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
        return [{"optimizer": disc_opt, "frequency": self.opt_hparams['disc_freq']},
                {"optimizer": gen_opt, "frequency": self.opt_hparams['gen_freq']}]

    def validation_step(self, batch, batch_idx):

        x, y = batch[self.cond_idx], batch[self.real_idx]

        preds = []
        for i in range(self.val_hparams['val_nens']):
            noise = torch.randn(y.shape[0], *self.noise_shape[-3:], device=self.device)
            pred, lr_corrected_pred = self.gen(x, noise)
            pred = pred.detach().to('cpu').numpy().squeeze()
            preds.append(pred)
        preds = np.array(preds)
        truth = y.detach().to('cpu').numpy().squeeze(1)
        truth = xr.DataArray(
            truth,
            dims=['sample', 'lat', 'lon'],
            name='tp'
        )

        #         print("preds shape", preds.shape)

        preds = xr.DataArray(
            preds,
            dims=['member', 'sample', 'lat', 'lon'],
            name='tp'
        )

        truth = truth * (self.val_hparams['ds_max'] - self.val_hparams['ds_min']) + self.val_hparams['ds_min']

        preds = preds * (self.val_hparams['ds_max'] - self.val_hparams['ds_min']) + self.val_hparams['ds_min']

        if self.val_hparams['tp_log']:
            truth = log_retrans(truth, self.val_hparams['tp_log'])
            preds = log_retrans(preds, self.val_hparams['tp_log'])

        crps = []
        rmse = []
        for sample in range(x.shape[0]):
            sample_crps = xs.crps_ensemble(truth.sel(sample=sample), preds.sel(sample=sample)).values
            sample_rmse = xs.rmse(preds.sel(sample=sample).mean('member'), truth.sel(sample=sample),
                                  dim=['lat', 'lon']).values
            crps.append(sample_crps)
            rmse.append(sample_rmse)

        crps = torch.tensor(np.mean(crps), device=self.device)
        rmse = torch.tensor(np.mean(rmse), device=self.device)
        self.log('val_crps', crps, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_rmse', rmse, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)

        return crps