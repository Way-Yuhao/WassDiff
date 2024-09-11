__author__ = 'yuhao liu'
from typing import Any, Dict, Optional, Tuple, List
from torch.utils.data._utils.collate import default_collate
import xarray as xr
from src.data.cpc_mrms_dataset import DailyAggregateRainfallDataset


class RainfallDatasetInference(DailyAggregateRainfallDataset):

    def __init__(self, data_config: dict,
                 dataloader_mode: Optional[str] = None,
                 specify_eval_targets: Optional[List[Dict[str, Any]]] = None):
        super().__init__(data_config)
        self.dataloader_mode = dataloader_mode
        self.specify_eval_targets = specify_eval_targets

        assert self.dataloader_mode in ['specify_eval', 'eval_set_random', 'eval_set_deterministic']
        return

    def get_item_specified(self, item: int):
        date_ = self.specify_eval_targets[item]['date']
        lon = self.specify_eval_targets[item]['lon']
        lat = self.specify_eval_targets[item]['lat']
        return super().get_arbitrary_item(date_, lon, lat)

    def __getitem__(self, item: int):
        if self.dataloader_mode == 'specify_eval':
            return self.get_item_specified(item)
        elif self.dataloader_mode == 'eval_set_random':
            raise NotImplementedError()
        else:
            raise NotImplementedError()


def xarray_collate_fn(batch):
    # Create a list for each batch element
    collated_batch = []
    for element in zip(*batch):
        # Check if element is an xarray.DataArray
        if isinstance(element[0], xr.DataArray):
            # Leave xarray.DataArray unchanged
            collated_batch.append(list(element))  # Retain each xarray object as is
        # Check if element is a tuple and the first element is a dict with xr.DataArray values
        elif isinstance(element[0], dict) and all(isinstance(v, xr.DataArray) for v in element[0].values()):
            # Leave the dict of xarray.DataArray unchanged
            collated_batch.append(list(element))  # Retain the dict unchanged

        else:
            # Use default collate for other types
            collated_batch.append(default_collate(element))
    return tuple(collated_batch)  # Return as a tuple of 4 elements
