# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from hydra import compose, initialize
from src.models.ncsn import utils as mutils
from src.utils.ncsn_utils.sde_lib import VESDE, VPSDE
# from util.helper import custom_summary
from torchinfo import summary
from src.utils.helper import alert


def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, t)

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False, use_emd=False, emd_weight: float = 0, compute_rescaled_emd=False,
                     emd_rescale_c=500):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition):  # original function
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device) # time labels?
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None, None, None]
        perturbed_data = noise + batch
        score = model_fn(perturbed_data, labels, condition)  # executes forward pass here
        target = -noise / (sigmas ** 2)[:, None, None, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
        loss = torch.mean(losses)
        return loss, {'sigmas': sigmas, 'noise': noise, 'score': score, 'target': target,
                      'perturbed_data': perturbed_data,
                      'error_map': torch.square(score - target),
                      'denoised_data': perturbed_data + (score * (sigmas ** 2)[:, None, None, None])}  # * sigmas[:, None, None, None], ?

    def loss_fun_score_emd(model, batch, condition):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)  # random time step
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None, None, None]
        perturbed_data = noise + batch

        # custom_summary(model, perturbed_data.shape[1:], labels.shape[1:], condition.shape[1:])
        # summary(model, perturbed_data, labels, condition)


        score = model_fn(perturbed_data, labels, condition)  # executes forward pass here
        target = -noise / (sigmas ** 2)[:, None, None, None]
        # compute loss on score
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
        score_loss = torch.mean(losses)
        # compute loss on emd
        denoised_data = (perturbed_data + (score * (sigmas ** 2)[:, None, None, None])).float()
        denoised_gt = perturbed_data + (target * (sigmas ** 2)[:, None, None, None])
        # convert denoised_data and denoised_gt to pytorch tensor of shape (batch_size, n_features)
        emd = sliced_wasserstein_distance(denoised_data.view(denoised_data.shape[0], -1),
                                   denoised_gt.view(denoised_gt.shape[0], -1),
                                          rescaled=compute_rescaled_emd, c=emd_rescale_c)

        # emd = sliced_wasserstein_distance(denoised_data.reshape(-1, 1),
        #                            denoised_gt.reshape(-1, 1),
        #                                   rescaled=compute_rescaled_emd, c=emd_rescale_c)
        # # check for nan, should only apply to .reshape(-1, 1)
        # if torch.isnan(emd):
        #     alert(f'Nan value detected in emd loss. Setting to 0')
        #     emd = 0

        # emd = emd_hist_sample(denoised_data, denoised_gt)

        # compute total loss
        loss = (1 - emd_weight) * score_loss + emd_weight * emd
        return loss, {'emd_loss': emd, 'score_loss': score_loss,
                      'sigmas': sigmas, 'noise': noise, 'score': score, 'target': target,
                      'perturbed_data': perturbed_data,
                      'error_map': torch.square(score - target),
                      'denoised_data': denoised_data, 'denoised_gt': denoised_gt,}

    return loss_fun_score_emd if use_emd else loss_fn


def emd_hist_sample(output_batch: torch.tensor, gt_batch: torch.tensor):
    # make sure the input is 4D tensor
    assert output_batch.dim() == 4 and gt_batch.dim() == 4
    # flatten the tensors
    output_batch = output_batch.view(output_batch.shape[0], -1)
    gt_batch = gt_batch.view(gt_batch.shape[0], -1)
    # compute the emd
    emds = []
    for i in range(output_batch.shape[0]):
        output = output_batch[i].flatten()
        gt = gt_batch[i].flatten()
        # compute the emd in pytorch
        proj_p, _ = torch.sort(output)
        proj_q, _ = torch.sort(gt)

        # Compute 1D Wasserstein distance (L1 norm between sorted projections)
        emds += [torch.mean(torch.abs(proj_p - proj_q))]
    emd = torch.mean(torch.stack(emds))
    return emd

def sliced_wasserstein_distance(p, q, num_projections=100, rescaled=False, c=500):
    """
    Compute the Sliced Wasserstein Distance between distributions.

    Args:
    p (Tensor): Source distribution points (batch_size, n_features).
    q (Tensor): Target distribution points (batch_size, n_features).
    num_projections (int): Number of random 1D projections to use.

    Returns:
    Tensor: Approximated Wasserstein distance.
    """
    if rescaled:  # rescale the data
        # cannot handel -1 values because of noise
        # invalid_regions = q == -1
        # p = p * c
        # q = q * c
        # p[invalid_regions] = -1
        # q[invalid_regions] = -1
        p = p * c
        q = q * c

    dim = p.shape[1]
    max_dist = 0
    for _ in range(num_projections):
        # Random projection vector
        theta = torch.randn(dim, device=p.device, dtype=p.dtype)
        theta /= theta.norm(p=2)  # Normalize

        # Project distributions onto theta
        proj_p = torch.matmul(p, theta)
        proj_q = torch.matmul(q, theta)

        # Sort projections
        proj_p, _ = torch.sort(proj_p)
        proj_q, _ = torch.sort(proj_q)

        # Compute 1D Wasserstein distance (L1 norm between sorted projections)
        dist = torch.mean(torch.abs(proj_p - proj_q))
        max_dist += dist

    return max_dist / num_projections


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                         sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        score = model_fn(perturbed_data, labels)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False,
                use_emd=False, emd_weight=0, compute_rescaled_emd=False, emd_rescale_c=500):
    """Create a one-step training/evaluation function.

      Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        optimize_fn: An optimization function.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to
          https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

      Returns:
        A one-step function for training or evaluation.
  """
    if continuous:
        loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                                  continuous=True, likelihood_weighting=likelihood_weighting)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean, use_emd=use_emd, emd_weight=emd_weight,
                                       compute_rescaled_emd=compute_rescaled_emd, emd_rescale_c=emd_rescale_c)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(state, batch, condition):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          state: A dictionary of training information, containing the score model, optimizer,
           EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data. (as GT?)

        Returns:
          loss: The average loss value of this state.
    """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss, loss_dict = loss_fn(model, batch, condition)  # executes forward pass here
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss, loss_dict = loss_fn(model, batch, condition)
                ema.restore(model.parameters())

        return loss, loss_dict

    return step_fn
