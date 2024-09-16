__author__ = 'yuhao liu'

import sys
import os
import traceback
import warnings
import functools
from typing import Dict, List, Tuple, Any, Union
import time
import datetime as dt
import rootutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import torch
from torchvision.utils import make_grid, save_image
from rich.progress import track, Progress, TextColumn, \
    BarColumn, TimeRemainingColumn, TaskProgressColumn, MofNCompleteColumn
import uuid
import wandb
from matplotlib.colors import Normalize
import cv2
import requests
from dotenv import load_dotenv
import contextlib
import io
from functools import wraps
from torchsummary import summary
from collections import OrderedDict
import torch.nn as nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

slack_alert_msg_printed = False
NUM_CLASSES = 6  # including y=0 (cloud)
VALIDATION_SPLIT = 0.2  # percentage of training data reserved for validation
VIS_PARAM = {
    'label_min': 1,
    'label_max': 5,
    'label_palette': [
        # implicitly: 0, Black
        [255, 255, 0],  # 1, Yellow
        [218, 165, 32],  # 2, Gold
        [0, 150, 0],  # 3, Dark Green
        [150, 255, 150],  # 4, Light Green
        [0, 0, 255],  # 5, Blue
    ]
}


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


####################################################
def yprint(msg):
    """
    Print to stdout console in yellow.
    :param msg:
    :return:
    """
    print(f"{bcolors.WARNING}{msg}{bcolors.ENDC}")


def rprint(msg):
    """
    Print to stdout console in red.
    :param msg:
    :return:
    """
    print(f"{bcolors.FAIL}{msg}{bcolors.ENDC}")


def pjoin(*args):
    """
    Joins paths for OS file system while ensuring the corrected slashes are used for Windows machines
    :param args:
    :return:
    """
    path = os.path.join(*args).replace("\\", "/")
    return path


def print_segment(nlines=1):
    print('---------------------------------')
    for _ in range(nlines):
        print('\n')


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def time_func(func):
    def inner(*args, **kwargs):
        start_time = time.monotonic()
        func(*args, **kwargs)
        yprint('---------------------------------')
        stop_time = time.monotonic()
        yprint(f'Processing time = {dt.timedelta(seconds=stop_time - start_time)}')

    return inner


class Suppressor(object):

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            raise

    def write(self, x): pass


def get_season(now):
    Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
    seasons = [('winter', (dt.date(Y, 1, 1), dt.date(Y, 3, 20))),
               ('spring', (dt.date(Y, 3, 21), dt.date(Y, 6, 20))),
               ('summer', (dt.date(Y, 6, 21), dt.date(Y, 9, 22))),
               ('autumn', (dt.date(Y, 9, 23), dt.date(Y, 12, 20))),
               ('winter', (dt.date(Y, 12, 21), dt.date(Y, 12, 31)))]
    # now = dt.datetime.strptime(str(now), '%Y%m%d').date()
    if isinstance(now, dt.datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)


def view_colormap(cmap_kw):
    """Plot a colormap with its grayscale equivalent"""

    def grayscale_cmap(cmap):
        """Return a grayscale version of the given colormap"""
        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))

        # convert RGBA to perceived grayscale luminance
        # cf. http://alienryderflex.com/hsp.html
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]

        return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

    cmap = plt.cm.get_cmap(cmap_kw)
    colors = cmap(np.arange(cmap.N))

    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots(2, figsize=(6, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
    plt.show()


def get_training_progressbar():
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        auto_refresh=True
    )
    return progress


class InfiniteLoader:
    """
    Wraps a DataLoader, reusing the iterator indefinitely.
    Intended to avoid StopIteration exceptions.
    """

    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = next(self.iterator)
        except StopIteration:
            # If StopIteration is raised, create a new iterator
            self.iterator = iter(self.loader)
            item = next(self.iterator)
        return item


def cm_(t, cmap_kw='magma', vmin=None, vmax=None):
    """
    Performs color map on a pytorch tensor. Returns a numpy array of shape [h, w, 3]
    """
    cmap = cm.get_cmap(cmap_kw)
    arr = torch.squeeze(t, dim=0).numpy()
    if vmin is not None and vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)
        arr = norm(arr)
    elif vmin is None and vmax is None:
        pass
    elif vmin is None or vmax is None:
        raise AttributeError("vmin and vmax must be both specified or both None")
    t_mapped = cmap(arr)[:, :, 0:3]
    return t_mapped
    # return Image.fromarray(np.uint8(cm.gist_earth(t_mapped) * 255))
    # return torch.from_numpy(t_mapped).unsqueeze(dim=0)


def view_colormap(cmap_kw):
    """Plot a colormap with its grayscale equivalent"""

    def grayscale_cmap(cmap):
        """Return a grayscale version of the given colormap"""
        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))

        # convert RGBA to perceived grayscale luminance
        # cf. http://alienryderflex.com/hsp.html
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]

        return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

    cmap = plt.cm.get_cmap(cmap_kw)
    colors = cmap(np.arange(cmap.N))

    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots(2, figsize=(6, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
    plt.show()


def hash_(str_=None, bits=4):
    """
    modifies output filename by appending a hashcode. If input (str_) is provided,
    it must contain substring '{}' as a placeholder for hashcode.
    """
    hashcode = uuid.uuid4().hex[:bits]
    if str_ is not None:
        return str_.format(hashcode)
    else:
        return hashcode


def visualize_batch(**kwargs):
    vis_param = {
        'surf_temp': 'viridis',
        'elevation': 'viridis',
        'vflux_e': 'RdBu',
        'vflux_n': 'RdBu',
        'wind_u': 'RdBu',
        'wind_v': 'RdBu',
        'precip_lr': 'magma',
        'precip_up': 'magma',
        'precip_gt': 'magma',
        'precip_hr': 'magma',
        'precip_masked': 'magma',
        'mask': 'binary',
        'density': 'viridis',
    }
    n = 5
    for k, v in kwargs.items():
        grid = make_grid(v[0:n, :, :, :])
        grid_mono = grid[0, :, :].unsqueeze(0)
        if 'precip' not in k:
            grid_mono /= grid_mono.max()
        cm_grid = cm_(grid_mono.detach().cpu(), vis_param[k])
        images = wandb.Image(cm_grid, caption=k)
        wandb.log({f"dataloader/{k}": images})


def visualize_batch_aggregate(**kwargs):
    raise NotImplementedError()


def display_config_warnings():
    raise NotImplementedError()  # TODO


def wandb_display_grid(img_tensor, log_key, caption, step, ncol, norm_factor=None):
    """
    Displays a grid of images in wandb.
    :param img_tensor: tensor of shape [n, c, h, w]
    :param log_key: key for logging to wandb
    :param caption: caption for the image
    :param step: step for logging to wandb
    :param ncol: number of columns in the grid. Assuming one row
    :param normalize_noise: if True, normalize noise by dividing by 50
    :return:
    """
    grid = make_grid(img_tensor[0:ncol, :, :, :])
    grid_mono = grid[0, :, :].unsqueeze(0)
    if norm_factor is not None:
        grid_mono /= norm_factor
    cm_grid = cm_(grid_mono.detach().cpu())
    images = wandb.Image(cm_grid, caption=caption)
    wandb.log({log_key: images}) #, step=step)


def alert(message):
    """
    Sends a message to a designated slack channel, which a SLACK_WEBHOOK_URL to be set in .env file.
    If webhook URL is not found, the message is printed to stdout in red.
    :param message:
    :return:
    """
    global slack_alert_msg_printed  # print out error msg only once
    path_to_restore = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    load_dotenv('.env')
    os.chdir(path_to_restore)
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    if webhook_url is None:
        if not slack_alert_msg_printed:
            msg = 'To send alerts to slack, set SLACK_WEBHOOK_URL in .env file under project root directory.'
            yprint(msg)  # Assuming yprint is a typo and meant print. Adjust as necessary for your logging method.
            yprint('Message routed to stdout')
            slack_alert_msg_printed = True  # Mark the warning as printed
        rprint(message)
        return
    else:
        data = {'text': message,
                'username': 'Webhook Alert',
                'icon_emoji': ':robot_face:'}
        response = requests.post(webhook_url, json=data)
        if response.status_code != 200:
            raise ValueError(
                'Request to slack returned an error %s, the response is:\n%s'
                % (response.status_code, response.text)
            )
        return


def monitor(func):
    """
    Decorator to monitor the execution of a function. If the function fails, the stack trace is printed.
    No message is sent at completion.
    :param func:
    :return:
    """
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            stack_trace = traceback.format_exc()
            title = 'Code failed'
            alert(f'*{title}*```{stack_trace}```')
            raise e
    return inner


def monitor_complete(func):
    """
    Decorator to monitor the execution of a function. If the function fails, the stack trace is printed.
    If the function succeeds, a message is sent to a designated slack channel.
    :param func:
    :return:
    """
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
            alert(f'Code executed successfully: {func.__name__}')
        except Exception as e:
            stack_trace = traceback.format_exc()
            title = 'Code failed'
            alert(f'*{title}*```{stack_trace}```')
            raise e
    return inner


def capture_stdout(func):
    """
    Suppresses stdout output, unless it contains an error message, in which case a ValueError is raised.
    :param func:
    :return:
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a StringIO buffer to capture output
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            # Call the original function and capture its return value
            retval = func(*args, **kwargs)
        # Retrieve the captured output
        std_output = stdout_capture.getvalue()
        if ('error' in std_output.lower() or 'warning' in std_output.lower()
                or 'failed' in std_output.lower() or 'exception' in std_output.lower()):
            raise ValueError(f'Found error in suppressed stdout from {func.__name__}\n{std_output}')
        # Return both the function's original return value and the captured output
        return retval
    return wrapper


# Define a custom summary function
def custom_summary(model, *inputs):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    summary = OrderedDict()
    hooks = []

    # Multiple inputs to the network
    inputs = [torch.rand(2, *in_size).to('cuda:0') for in_size in inputs]
    batch_size = inputs[0].size(0)

    model.apply(register_hook)
    model(*inputs)

    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    print("Layer (type:depth-idx)               Output Shape         Param #")
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        line_new = "{:>50}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    total_input_size = abs(np.prod(sum(inputs, torch.Size())) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % (total_input_size + total_output_size + total_params_size))
    print("----------------------------------------------------------------")


def move_batch_to_cpu(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Moves a dict of tensors from a cuda device to cpu.
    """
    return {key: tensor.cpu() for key, tensor in batch.items()}


def extract_values_from_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extracts values from a dict of tensors. Expects each tensor to be of shape [1,]
    """
    return {key: tensor.item() for key, tensor in batch.items()}


def squeeze_batch(batch: Dict[str, torch.Tensor], dim: Union[int, Tuple]) -> Dict[str, torch.Tensor]:
    """
    Squeezes a dict of tensors.
    """
    return {key: torch.squeeze(tensor, dim=dim) for key, tensor in batch.items()}


if __name__ == '__main__':
    a = torch.ones(1, 1, 128, 128)
    b = torch.ones(1, 1, 128, 128)