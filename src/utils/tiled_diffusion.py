import torch
from src.utils.image_spliter import ImageSpliterTh

class TiledDiffusion:
    def __init__(self, tiled_config: dict):
        """
        Initialize TiledDiffusion with tiling parameters.

        Args:
            tiled_config (dict): A dictionary with keys such as:
                - patch_size (int): Default 256.
                - stride (int): Default 192.
                - scale_factor (int): Default 1.
                - batch_size (int): Default 12.
        """
        self.patch_size = tiled_config['patch_size']
        self.stride = tiled_config["stride"]
        self.sf = tiled_config["sf"]
        self.batch_size = tiled_config["batch_size"]

    def sample(self, x, timesteps, noise, smld_sigma_array,
               *, model, condition, w, out_dim, save_dir, null_condition,
               display_pbar, gt, null, corrector_fn, projector_fn):
        """
        Run tiled (patch-based) diffusion sampling.

        Args:
            x (Tensor): Initial full-image sample.
            timesteps (Tensor): Precomputed timesteps.
            noise (Tensor): Clone of initial sample (unused in this snippet, but available).
            smld_sigma_array (Tensor): Precomputed sigma array.
            model: The diffusion model.
            condition (Tensor): Conditioning tensor.
            w: Guidance weight.
            out_dim: Output dimensions.
            save_dir: Optional directory to save outputs.
            null_condition (Tensor): Null condition.
            display_pbar (bool): Whether to display a progress bar.
            gt: Optional ground truth tensor.
            null: Null token value.
            corrector_fn: Function to perform the corrector update.
            projector_fn: Function to perform the projector update.

        Returns:
            Tensor: The final upsampled image (after applying inverse scaling).
        """
        # Set up a progress iterator if requested.
        iterator = range(len(timesteps))
        if display_pbar:
            from rich.progress import track
            iterator = track(range(len(timesteps)),
                             description=f'Tiled Sampling {len(timesteps)} steps...',
                             refresh_per_second=1)

        for i in iterator:
            t = timesteps[i]
            # Create a new splitter for the current full image.
            im_spliter = ImageSpliterTh(x, self.patch_size, self.stride, sf=self.sf)
            # Prepare lists for patch accumulation.
            pch_x_list = []
            pch_condition_list = []
            pch_null_condition_list = []
            idx_info_list = []

            # Iterate over patches produced by the splitter.
            for pch_x, idx_info in im_spliter:
                h_start_sf, h_end_sf, w_start_sf, w_end_sf = idx_info
                # Determine corresponding indices in the condition.
                h_start = h_start_sf // self.sf
                h_end = h_end_sf // self.sf
                w_start = w_start_sf // self.sf
                w_end = w_end_sf // self.sf
                pch_condition = condition[:, :, h_start:h_end, w_start:w_end]

                pch_x_list.append(pch_x)
                pch_condition_list.append(pch_condition)
                if null_condition is not None:
                    pch_null_condition = null_condition[:, :, h_start:h_end, w_start:w_end]
                    pch_null_condition_list.append(pch_null_condition)
                idx_info_list.append(idx_info)

                # Process in batches.
                if len(pch_x_list) == self.batch_size:
                    batch_pch_x = torch.cat(pch_x_list, dim=0)
                    batch_pch_condition = torch.cat(pch_condition_list, dim=0)
                    batch_pch_null_condition = (torch.cat(pch_null_condition_list, dim=0)
                                                if pch_null_condition_list else None)

                    batch_pch_x, batch_pch_x_mean = corrector_fn(
                        model, x=batch_pch_x, t=t, c=batch_pch_condition, w=w,
                        null_cond=batch_pch_null_condition)
                    batch_pch_x, batch_pch_x_mean = projector_fn(
                        model, x=batch_pch_x, t=t, c=batch_pch_condition, w=w,
                        null_cond=batch_pch_null_condition)

                    # Update the full image with the processed patches.
                    for j, info in enumerate(idx_info_list):
                        patch = batch_pch_x_mean[j:j + 1] if i == len(timesteps) - 1 else batch_pch_x[j:j + 1]
                        im_spliter.update_gaussian(patch, info)

                    # Reset batch lists.
                    pch_x_list, pch_condition_list = [], []
                    pch_null_condition_list, idx_info_list = [], []

            # Process any remaining patches.
            if len(pch_x_list) > 0 and len(pch_x_list) < self.batch_size:
                batch_pch_x = torch.cat(pch_x_list, dim=0)
                batch_pch_condition = torch.cat(pch_condition_list, dim=0)
                batch_pch_null_condition = (torch.cat(pch_null_condition_list, dim=0)
                                            if pch_null_condition_list else None)
                batch_pch_x, batch_pch_x_mean = corrector_fn(
                    model, x=batch_pch_x, t=t, c=batch_pch_condition, w=w,
                    null_cond=batch_pch_null_condition)
                batch_pch_x, batch_pch_x_mean = projector_fn(
                    model, x=batch_pch_x, t=t, c=batch_pch_condition, w=w,
                    null_cond=batch_pch_null_condition)
                for j, info in enumerate(idx_info_list):
                    patch = batch_pch_x_mean[j:j + 1] if i == len(timesteps) - 1 else batch_pch_x[j:j + 1]
                    im_spliter.update_gaussian(patch, info)

            # Gather the updated full image.
            x = im_spliter.gather()

        return x
