"""
Metrics for evaluating a single output against gt. Requires output and gt in numpy format.
"""
__author__ = 'yuhao liu'

import os
import numpy as np
from skimage.measure import block_reduce
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
import torch
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from src.utils.pysteps.spatialscores import fss
from src.utils.pysteps.probscores import CRPS
from scipy.linalg import sqrtm
# from hydra import initialize, compose
# from hydra.utils import call
import xarray as xr
import pandas as pd
import lpips
from src.utils.helper import time_func, monitor_complete, deprecated

# constants
# with initialize(version_base=None, config_path="../configs"):
#     # Compose the configuration using the config name and any overrides you wish to apply
#     cfg = compose(config_name="downscale_cpc_density")
# mppe_thres = cfg.eval.peak_mesoscale_threshold
# hrre_thres = cfg.eval.heavy_rain_threshold


def scale_logp1(precip):
    precip = precip.copy()
    valid_region = precip != -1
    scaled = np.zeros_like(precip)
    scaled[valid_region] = np.log(precip[valid_region] + 1)
    normalized = scaled / 5.
    normalized[~valid_region] = -1
    return normalized


def apply_valid_mask(*args, valid_mask: np.ndarray):
    """
    Apply valid mask to all input arrays.
    Returns a list of 1D arrays.
    """
    if valid_mask is not None:
        return [arg[valid_mask] for arg in args]
    else:
        return args


def pooling(*args, k: int, func: str = 'max'):
    """
    Pooling function for spatial pooling. Currently supports max and mean pooling.
    The array of type bool (assumed to be valid mask) will be pooled using min function,
    meaning that any False (invalid) value in the block will result in False in the pooled block.
    :param args: list of arrays to be pooled
    :param k: pooling size
    :param func: pooling function, 'max' or 'mean'
    :return: list of pooled arrays
    """
    assert func in ['max', 'mean'], f"Pooling function {func} not supported."
    if k == 1:
        return args
    elif k <= 0:
        raise ValueError("Pooling size k should be greater than 0.")
    if func == 'max':
        func = np.max
    elif func == 'mean':
        func = np.mean
    pooled_args = []
    for arg in args:
        if arg.dtype == bool:
            pooled_arg = block_reduce(arg, block_size=(k, k), func=np.min).astype(bool)
        else:
            pooled_arg = block_reduce(arg, block_size=(k, k), func=func)
        pooled_args.append(pooled_arg)
    return pooled_args


def calc_bias(output: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray = None, k: int = 1, pooling_func='mean'):
    output, gt, valid_mask = pooling(output, gt, valid_mask, k=k, func=pooling_func)
    output, gt = apply_valid_mask(output, gt, valid_mask=valid_mask)
    return np.mean(output - gt)


def calc_mae(output: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray = None, k: int = 1, pooling_func='mean'):
    output, gt, valid_mask = pooling(output, gt, valid_mask, k=k, func=pooling_func)
    output, gt = apply_valid_mask(output, gt, valid_mask=valid_mask)
    return np.mean(np.abs(output - gt))


def calc_mse(output: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray = None, k: int = 1, pooling_func='mean'):
    output, gt, valid_mask = pooling(output, gt, valid_mask, k=k, func=pooling_func)
    output, gt = apply_valid_mask(output, gt, valid_mask=valid_mask)
    return np.mean((output - gt) ** 2)


def calc_rmse(output: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray = None, k: int = 1, pooling_func='mean'):
    output, gt, valid_mask = pooling(output, gt, valid_mask, k=k, func=pooling_func)
    output, gt = apply_valid_mask(output, gt, valid_mask=valid_mask)
    return np.sqrt(np.mean((output - gt) ** 2))


def calc_pcc(output: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray = None, k: int = 1, pooling_func='mean'):
    """
    Pearson Correlation Coefficient
    """
    output, gt, valid_mask = pooling(output, gt, valid_mask, k=k, func=pooling_func)
    output, gt = apply_valid_mask(output, gt, valid_mask=valid_mask)
    return np.corrcoef(output.flatten(), gt.flatten())[0, 1]


def calc_csi(output: np.ndarray, gt: np.ndarray, threshold: float = 10, valid_mask: np.ndarray = None, k: int = 1,
             pooling_func='mean'):
    """
    Critical Success Index
    :param output:
    :param gt:
    :param threshold:
    :return:
    """
    output, gt, valid_mask = pooling(output, gt, valid_mask, k=k, func=pooling_func)
    output, gt = apply_valid_mask(output, gt, valid_mask=valid_mask)
    b_output = output.copy()
    b_gt = gt.copy()
    b_output[output < threshold] = 0
    b_output[output >= threshold] = 1
    b_gt[gt < threshold] = 0
    b_gt[gt >= threshold] = 1
    # csi
    tp = np.sum(np.logical_and(b_output == 1, b_gt == 1))
    fp = np.sum(np.logical_and(b_output == 1, b_gt == 0))
    fn = np.sum(np.logical_and(b_output == 0, b_gt == 1))
    if (tp + fp + fn) == 0:
        return np.nan
    else:
        csi = tp / (tp + fp + fn)
        return csi


def calc_ssim(output: np.ndarray, gt: np.ndarray):

    # if valid_mask is not None:
    #     output = output[valid_mask]
    #     gt = gt[valid_mask]
    # return ssim(output, gt, data_range=gt.max() - gt.min())

    # scale the input to 0-1
    output = scale_logp1(output)
    gt = scale_logp1(gt)
    # output[output == -1] = 0
    # gt[gt == -1] = 0

    return ssim(output, gt, data_range=2.5)


def calc_emd(output: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray = None):
    """
    Earth Mover's Distance
    """
    output, gt = apply_valid_mask(output, gt, valid_mask=valid_mask)
    return wasserstein_distance(output, gt)


def calc_fss(output: np.ndarray, gt: np.ndarray, threshold: float = 10, k: int = 10):
    """
    Fractions Skill Score
    :param output:
    :param gt:
    :param threshold:
    :param k:
    :return:
    """
    if gt.max() < threshold:
        return np.nan
    return fss(output, gt, thr=threshold, scale=k)


def numpy_to_tensor(images, divide_img_by_c: float = 1.0):
    images = torch.from_numpy(images.copy()).float()
    images = images.permute(2, 0, 1)
    images /= divide_img_by_c
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((299, 299), antialias=True),  # Resize image to fit InceptionV3
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(images)
    return image.unsqueeze(0)
    # return torch.stack([transform(img) for img in images])


def extract_features(images, device, model=None):
    if model is None:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        model.fc = torch.nn.Identity()
        model = model.to(device)
        model.eval()

    with torch.no_grad():
        images = images.to(device)
        features = model(images)
    return features.cpu().numpy()


def calculate_fid(real_features, gen_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    if real_features.shape[0] == 1:
        sigma1 = sigma1.reshape((1, 1))
        sigma2 = sigma2.reshape((1, 1))

    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calc_fid(gen_images_np, real_images_np, rescale_to_1: bool = False, inception_model=None):
    if rescale_to_1 and real_images_np.max() > 1.0:
        c = real_images_np.max()
    else:
        c = 1.0
    # monochrome images
    if len(real_images_np.shape) == 2:
        # repeat 3 channels
        real_images_np = np.repeat(real_images_np[:, :, np.newaxis], 3, axis=2)
        gen_images_np = np.repeat(gen_images_np[:, :, np.newaxis], 3, axis=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert numpy arrays of images to PyTorch tensors and preprocess
    real_images = numpy_to_tensor(real_images_np, divide_img_by_c=c)
    gen_images = numpy_to_tensor(gen_images_np, divide_img_by_c=c)

    # Extract features
    real_features = extract_features(real_images, device, inception_model)
    gen_features = extract_features(gen_images, device, inception_model)

    # Calculate FID
    fid_score = calculate_fid(real_features, gen_features)
    return fid_score


def calc_hrre(output: np.ndarray, gt: np.ndarray, hrre_thres: float):
    """
    Heavy rain region error (HRRE) from RainNet
    # TODO
    """
    # output = output[output >= hrre_thres]
    # gt = gt[gt >= hrre_thres]
    # count number of pixels in heavy rain region
    output_count = np.sum(output >= hrre_thres)
    gt_count = np.sum(gt >= hrre_thres)
    hrre = np.abs(output_count - gt_count)
    return hrre


# def calc_mppe(output: np.ndarray, gt: np.ndarray):
#     """
#     Mesoscale peak precipitation error (MPPE), from RainNet
#     # TODO: concern: boundary issue?
#     """
#     # apply threshold
#     if output.max() < mppe_thres and gt.max() < mppe_thres:
#         return np.nan
#     output = output[output >= mppe_thres]
#     gt = gt[gt >= mppe_thres]
#     mppe = calc_rmse(output, gt, valid_mask=None)
#     return mppe

def calc_mppe(output: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray = None):
    """
    Mesoscale peak precipitation error (MPPE), from RainNet
    """
    output, gt = apply_valid_mask(output, gt, valid_mask=valid_mask)
    # compute 99.9% percentile of output
    output_999p = np.percentile(output, 99.9)
    gt_999p = np.percentile(gt, 99.9)
    mppe = np.sqrt(np.mean((output_999p - gt_999p) ** 2))
    return mppe


@deprecated
@time_func
@monitor_complete
def compute_99p9_percentile(mrms_dir: str):
    """
    Compute the 99.9th percentile of MRMS data
    """
    # List all NetCDF files in the directory
    files = [os.path.join(mrms_dir, f) for f in os.listdir(mrms_dir) if f.endswith('.nc')]
    print(f"Found {len(files)} files in {mrms_dir}\nLoading dataset...")
    dates = [pd.to_datetime(f.split('_')[-1].split('.')[0], format='%Y%m%d') for f in files]
    datasets = [xr.open_dataset(f).expand_dims(time=[date]) for f, date in zip(files, dates)]
    ds = xr.concat(datasets, dim='time')
    print('Opened dataset')
    quantile_value = ds.quantile(0.999, dim=["time", "lat", "lon"], numeric_only=True, skipna=True)  # Adjust dimensions if necessary
    # Force computation if using Dask
    quantile_value = quantile_value.compute()
    print("99.9th Percentile of Daily Precipitation:", quantile_value.values)


def calc_lpips(output: np.ndarray, gt: np.ndarray, lpips_model: lpips.LPIPS, device: str = 'cuda'):
    """
    Compute LPIPS metric
    """
    # Normalize images
    # output = (output - gt.min()) / (gt.max() - gt.min())
    # gt = (gt - gt.min()) / (gt.max() - gt.min())

    output = scale_logp1(output) - .5
    gt = scale_logp1(gt) - .5

    # Convert to tensor
    output = torch.from_numpy(output).unsqueeze(0).unsqueeze(0).float()
    gt = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float()
    # put to device, if necessary
    output = output.to(device)
    gt = gt.to(device)
    # Calculate LPIPS
    lpips_score = lpips_model.forward(output, gt)
    return lpips_score.item()

############# BLOW IS ENSEMBLE METRICS ####################


def calc_crps(output: np.ndarray, gt: np.ndarray):
    """
    Compute CRPS (Continuous Ranked Probability Score)
    """
    return CRPS(output, gt)


if __name__ == '__main__':
    compute_99p9_percentile(cfg.data.dataset_path.mrms_daily)