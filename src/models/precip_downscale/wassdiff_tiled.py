from typing import Any, Dict, Tuple, Optional
# import wandb
import torch
from src.models.precip_downscale.wassdiff import WassDiffLitModule

# This is a class used to generate tiled_diffusion using Wassdiff Model.
class WassDiffTiledLitModule(WassDiffLitModule):
    """
    This is a module for generating tiled diffusion images using the Wassdiff model.

    This class modifies the image size in the model configuration temporarily to match
    the tiled configuration during initialization. This allows the base model to be
    initialized with dimensions optimized for tiled processing. After initialization,
    it restores the original full canvas image size, preserving the capability to work
    with the complete image later on.
    """

    def __init__(self, model_config: dict, optimizer_config: dict, tiled_config: dict, tiled_diffusion: bool,
                 compile: bool,
                  *args, **kwargs) -> None:
        """
        Initialize a `WassDiffTiledLitModule` instance with tiled diffusion capabilities.
        Parameters:
            model_config (dict): A dictionary containing the model configuration parameters.
                Expected to have an attribute 'data.image_size' specifying the original image size.
            optimizer_config (dict): A dictionary containing the optimizer configuration parameters.
            tiled_config (dict): A dictionary containing the tiled diffusion configuration.
                Must include a 'model_size' key for the temporary image size.
            tiled_diffusion (bool): Indicates whether tiled diffusion should be used.
            compile (bool): Flag to determine if the model should be compiled during initialization.
            *args: Additional positional arguments for the parent class constructor.
            **kwargs: Additional keyword arguments for the parent class constructor.
        """
        original_image_size = model_config.data.image_size
        # initialize model with model_size
        model_config.data.image_size = tiled_config.model_size
        super().__init__(model_config, optimizer_config, compile, *args, **kwargs)
        # Restore the original full canvas image size for further usage
        model_config.data.image_size = original_image_size
        self.tiled_config = tiled_config
        self.tiled_diffusion = tiled_diffusion
        return

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

        # Determine output dimensions based on image_size type.
        if not isinstance(self.model_config.data.image_size, int):
            h = condition.shape[2]
            w=  condition.shape[3]
        else:
            h = w = self.model_config.data.image_size

        if self.hparams.bypass_sampling:
            batch_dict['precip_output'] = torch.zeros_like(gt)
        else:
            x = self.pc_upsampler(
                model=self.net,
                condition=self.scaler(condition),
                w=self.model_config.model.w_guide,
                out_dim=(batch_size, 1, h, w),
                save_dir=None,
                null_condition=null_condition,
                display_pbar=self.hparams.display_sampling_pbar,
                gt=gt,
                null=self.model_config.model.null_token,
                tiled_diffusion=self.tiled_diffusion,  # new config parameter
                tiled_config=self.tiled_config  # new config parameter
            )
            if self.hparams.num_samples == 1:
                    batch_dict['precip_output'] = x
            else:
                for i in range(self.hparams.num_samples):
                    batch_dict['precip_output_' + str(i)] = x[i, :, :, :]
                # print(batch_dict)
                # print(batch_coords)
                # print(xr_low_res_batch)
                # print(valid_mask)
        return {'batch_dict': batch_dict, 'batch_coords': batch_coords, 'xr_low_res_batch': xr_low_res_batch,
                'valid_mask': valid_mask}

if __name__ == "__main__":
    _ = WassDiffTiledLitModule(None, None, None, None)
