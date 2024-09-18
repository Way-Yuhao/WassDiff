__author__ = 'yuhao liu'

import os
import os.path as p
from typing import Any, Dict, Optional, Tuple, List
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Dataset
import xarray as xr
from natsort import natsorted
from src.data.cpc_mrms_dataset import DailyAggregateRainfallDataset


class RainfallSpecifiedInference(DailyAggregateRainfallDataset):
    """
    Precipitation dataset for inference, where the targets are specified by date, lon, and lat.
    """

    def __init__(self, data_config: dict,
                 specify_eval_targets: Optional[List[Dict[str, Any]]] = None):
        super().__init__(data_config)
        self.specify_eval_targets = specify_eval_targets

    def __len__(self):
        return len(self.specify_eval_targets)

    def __getitem__(self, item: int):
        date_ = self.specify_eval_targets[item]['date']
        lon = self.specify_eval_targets[item]['lon']
        lat = self.specify_eval_targets[item]['lat']
        return super().get_arbitrary_item(date_, lon, lat)


class PreSavedPrecipDataset(DailyAggregateRainfallDataset):
    """
    Pre-sampled set of crops of precipitation, stored in torch tensors. Use this for deterministic evaluation.
    """

    def __init__(self, data_config: dict, sample_path: str, stop_at_batch: Optional[int]):
        super().__init__(data_config)
        self.sample_path = sample_path
        self.stop_at_batch = stop_at_batch  # stop at this number of batches

        samples = os.listdir(self.sample_path)
        self.samples = natsorted([f for f in samples if f.endswith('.pt') and f.startswith('batch')])
        if self.stop_at_batch:
            self.samples = self.samples[:self.stop_at_batch]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item: int):
        batch_dict = torch.load(p.join(self.sample_path, self.samples[item]))
        batch_idx = {'batch_idx': item}
        return batch_dict, batch_idx, None, None


def xarray_collate_fn(batch):
    # Create a list for each batch element
    collated_batch = []
    for element in zip(*batch):
        # Check if element is an xarray.DataArray
        if isinstance(element[0], xr.DataArray):
            # Leave xarray.DataArray unchanged
            collated_batch.append(element[0])  # Retain each xarray object as is
        # Check if element is a tuple and the first element is a dict with xr.DataArray values
        elif isinstance(element[0], dict) and all(isinstance(v, xr.DataArray) for v in element[0].values()):
            # Leave the dict of xarray.DataArray unchanged
            collated_batch.append(element[0])  # Retain the dict unchanged

        else:
            # Use default collate for other types
            collated_batch.append(default_collate(element))
    return tuple(collated_batch)  # Return as a tuple of 4 elements


def do_nothing_collate_fn(batch):
    """
    Handles a dict of tensors; will not add a minibatch dimension
    """
    return batch[0]
