from src.models.ncsn import utils as mutils
import torch
import numpy as np
from matplotlib import pyplot as plt
from src.utils.ncsn_utils.sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
import functools
from rich.progress import track
from src.utils.tiled_diffusion import TiledDiffusion  # ensure this module is available in your project
from src.utils.image_spliter import ImageSpliterTh
# from src.utils.metrics import calc_emd
# from losses import sliced_wasserstein_distance

# global
show_vis_at_steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 980, 990, 999, 1000]


def get_pc_inpainter(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
    """Create an image inpainting function that uses PC samplers.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
      corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for the corrector.
      n_steps: An integer. The number of corrector steps per update of the corrector.
      probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
      continuous: `True` indicates that the score-based model was trained with continuous time.
      denoise: If `True`, add one-step denoising to final samples.
      eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

    Returns:
      An inpainting function.
    """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def get_inpaint_update_fn(update_fn):
        """Modify the update function of predictor & corrector to incorporate data information."""

        def inpaint_update_fn(model, data, mask, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                masked_data_mean, std = sde.marginal_prob(data, vec_t)
                masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
                x = x * (1. - mask) + masked_data * mask
                x_mean = x * (1. - mask) + masked_data_mean * mask
                return x, x_mean, masked_data_mean

        return inpaint_update_fn

    projector_inpaint_update_fn = get_inpaint_update_fn(predictor_update_fn)
    corrector_inpaint_update_fn = get_inpaint_update_fn(corrector_update_fn)

    def pc_inpainter(model, data, mask):
        """Predictor-Corrector (PC) sampler for image inpainting.

        Args:
          model: A score model.
          data: A PyTorch tensor that represents a mini-batch of images to inpaint.
          mask: A 0-1 tensor with the same shape of `data`. Value `1` marks known pixels,
            and value `0` marks pixels that require inpainting.

        Returns:
          Inpainted (complete) images.
        """
        with torch.no_grad():
            # Initial sample
            x = data * mask + sde.prior_sampling(data.shape).to(data.device) * (1. - mask)
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in range(sde.N):
                t = timesteps[i]
                x, x_mean, masked_data = corrector_inpaint_update_fn(model, data, mask, x, t)
                x, x_mean, masked_data = projector_inpaint_update_fn(model, data, mask, x, t)

                if i % 100 == 0: # show diffusion progression
                    plt.imshow(x[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
                    plt.title('masked data, i = %d' % i)
                    plt.show()
                    plt.close()

            return inverse_scaler(x_mean if denoise else x)

    return pc_inpainter


def get_pc_colorizer(sde, predictor, corrector, inverse_scaler,
                     snr, n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
    """Create a image colorization function based on Predictor-Corrector (PC) sampling.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
      corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for correctors.
      n_steps: An integer. The number of corrector steps per update of the predictor.
      probability_flow: If `True`, solve the probability flow ODE for sampling with the predictor.
      continuous: `True` indicates that the score-based model was trained with continuous time steps.
      denoise: If `True`, add one-step denoising to final samples.
      eps: A `float` number. The SDE/ODE will start from `eps` to avoid numerical stabilities.

    Returns: A colorization function.
    """

    # `M` is an orthonormal matrix to decouple image space to a latent space where the gray-scale image
    # occupies a separate channel
    M = torch.tensor([[5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
                      [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
                      [5.7735026e-01, 4.0824822e-01, -7.0710683e-01]])
    # `invM` is the inverse transformation of `M`
    invM = torch.inverse(M)

    # Decouple a gray-scale image with `M`
    def decouple(inputs):
        return torch.einsum('bihw,ij->bjhw', inputs, M.to(inputs.device))

    # The inverse function to `decouple`.
    def couple(inputs):
        return torch.einsum('bihw,ij->bjhw', inputs, invM.to(inputs.device))

    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def get_colorization_update_fn(update_fn):
        """Modify update functions of predictor & corrector to incorporate information of gray-scale images."""

        def colorization_update_fn(model, gray_scale_img, x, t):
            mask = get_mask(x)
            vec_t = torch.ones(x.shape[0], device=x.device) * t
            x, x_mean = update_fn(x, vec_t, model=model)
            masked_data_mean, std = sde.marginal_prob(decouple(gray_scale_img), vec_t)
            masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
            x = couple(decouple(x) * (1. - mask) + masked_data * mask)
            x_mean = couple(decouple(x) * (1. - mask) + masked_data_mean * mask)
            return x, x_mean

        return colorization_update_fn

    def get_mask(image):
        mask = torch.cat([torch.ones_like(image[:, :1, ...]),
                          torch.zeros_like(image[:, 1:, ...])], dim=1)
        return mask

    predictor_colorize_update_fn = get_colorization_update_fn(predictor_update_fn)
    corrector_colorize_update_fn = get_colorization_update_fn(corrector_update_fn)

    def pc_colorizer(model, gray_scale_img):
        """Colorize gray-scale images using Predictor-Corrector (PC) sampler.

        Args:
          model: A score model.
          gray_scale_img: A minibatch of gray-scale images. Their R,G,B channels have same values.

        Returns:
          Colorized images.
        """
        with torch.no_grad():
            shape = gray_scale_img.shape
            mask = get_mask(gray_scale_img)
            # Initial sample
            x = couple(decouple(gray_scale_img) * mask + \
                       decouple(sde.prior_sampling(shape).to(gray_scale_img.device)
                                * (1. - mask)))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in range(sde.N):
                t = timesteps[i]
                x, x_mean = corrector_colorize_update_fn(model, gray_scale_img, x, t)
                x, x_mean = predictor_colorize_update_fn(model, gray_scale_img, x, t)

                # if i % 50 == 0: # show diffusion progression
                #     plt.imshow(x[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
                #     plt.title(i)
                #     plt.show()
                #     plt.close()

            return inverse_scaler(x_mean if denoise else x)

    return pc_colorizer


def get_pc_upsampler(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
    """Create an image super-resolution function that uses PC samplers.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
      corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for the corrector.
      n_steps: An integer. The number of corrector steps per update of the corrector.
      probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
      continuous: `True` indicates that the score-based model was trained with continuous time.
      denoise: If `True`, add one-step denoising to final samples.
      eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

    Returns:
      An super-resolution function.
    """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def get_upsample_update_fn(update_fn):
        """Modify the update function of predictor & corrector to incorporate data information."""

        def upsample_update_fn(model, low_res_img, x, t):
            with torch.no_grad():
                vec_t = torch.ones(low_res_img.shape[0], device=low_res_img.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                low_res_mean, std = sde.marginal_prob(low_res_img, vec_t)
                low_res_data = low_res_mean + torch.randn_like(x) * std[:, None, None, None]

                if t < .4: # .25 for 4x, .3 for 8x
                    t = 0

                x = x * (1 - t) + low_res_data * t
                x_mean = x * (1 - t) + low_res_mean * t
                return x, x_mean

        return upsample_update_fn

    projector_upsample_update_fn = get_upsample_update_fn(predictor_update_fn)
    corrector_upsample_update_fn = get_upsample_update_fn(corrector_update_fn)

    def pc_upsampler(model, low_res_img):
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(low_res_img.shape).to(low_res_img.device)
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in track(range(sde.N), description=f'Sampling {sde.N} steps....', refresh_per_second=1):
                t = timesteps[i]
                x, x_mean = corrector_upsample_update_fn(model, low_res_img, x, t)
                x, x_mean = projector_upsample_update_fn(model, low_res_img, x, t)

                # if i % 200 == 0:  # show diffusion progression
                #     plt.imshow(x[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
                #     plt.title('masked data, i = %d' % i)
                #     plt.show()
                #     plt.close()

            return inverse_scaler(x_mean if denoise else x)

    return pc_upsampler

def get_pc_cfg_upsampler(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
    """Create an image super-resolution function that uses PC samplers.
        Mimics the sampling scheme of classifier-free guidance

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
      corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for the corrector.
      n_steps: An integer. The number of corrector steps per update of the corrector.
      probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
      continuous: `True` indicates that the score-based model was trained with continuous time.
      denoise: If `True`, add one-step denoising to final samples.
      eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

    Returns:
      An super-resolution function.
    """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def get_upsample_update_fn(update_fn):
        """Modify the update function of predictor & corrector to incorporate data information."""

        def upsample_update_fn(model, x, t, c, w, null_cond):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model, c=c, w=w, null_cond=null_cond)

                return x, x_mean

        return upsample_update_fn

    projector_upsample_update_fn = get_upsample_update_fn(predictor_update_fn)
    corrector_upsample_update_fn = get_upsample_update_fn(corrector_update_fn)

    def pc_upsampler(model, condition, w, out_dim, save_dir=None, null_condition=None,
                         display_pbar=True, gt=None, null=None, tiled_diffusion=False, tiled_config=None):
            # Precompute sigma arrays and initialize the sample.
            discrete_sigmas = torch.exp(torch.linspace(np.log(0.01), np.log(50), 1000))
            smld_sigma_array = torch.flip(discrete_sigmas, dims=(0,))
            with torch.no_grad():
                x = sde.prior_sampling(out_dim).to(condition.device)
                noise = x.clone()
                timesteps = torch.linspace(sde.T, eps, sde.N)

                # If tiled diffusion is enabled, call the helper class.
                if tiled_diffusion and tiled_config is not None:
                    # Pass the precomputed objects along with the update functions.
                    tiled_diffuser = TiledDiffusion(tiled_config)
                    x = tiled_diffuser.sample(
                        x, timesteps, noise, smld_sigma_array,
                        model=model,
                        condition=condition,
                        w=w,
                        out_dim=out_dim,
                        save_dir=save_dir,
                        null_condition=null_condition,
                        display_pbar=display_pbar,
                        gt=gt,
                        null=null,
                        corrector_fn=corrector_upsample_update_fn,
                        projector_fn=projector_upsample_update_fn
                    )
                    return inverse_scaler(x)

                # Otherwise, run the original full-image diffusion loop.
                if display_pbar:
                    for i in track(range(sde.N), description=f'Sampling {sde.N} steps....', refresh_per_second=1):
                        t = timesteps[i]
                        x, x_mean = corrector_upsample_update_fn(model, x=x, t=t, c=condition, w=w,
                                                                 null_cond=null_condition)
                        x, x_mean = projector_upsample_update_fn(model, x=x, t=t, c=condition, w=w,
                                                                 null_cond=null_condition)
                        # (Optional: visualization code could go here)
                else:
                    for i in range(sde.N):
                        t = timesteps[i]
                        x, x_mean = corrector_upsample_update_fn(model, x=x, t=t, c=condition, w=w,
                                                                 null_cond=null_condition)
                        x, x_mean = projector_upsample_update_fn(model, x=x, t=t, c=condition, w=w,
                                                                 null_cond=null_condition)

                return inverse_scaler(x_mean if denoise else x)

    return pc_upsampler