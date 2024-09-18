from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, SubsetRandomSampler
from hydra import compose, initialize
from src.data.cpc_mrms_dataset import DailyAggregateRainfallDataset
from src.data.precip_dataloader_inference import (RainfallSpecifiedInference, PreSavedPrecipDataset,
                                                  xarray_collate_fn, do_nothing_collate_fn)


class PrecipDataModule(LightningDataModule):
    """`LightningDataModule` for the precipitation dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(self, data_config: dict, batch_size: int = 12, num_workers: int = 1, pin_memory: bool = False,
                 seed: int = 42, dataloader_mode: str = 'train', *args, **kwargs) -> None:
        """Initialize a `PrecipDataModule`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # allows to access init params with 'self.hparams' attribute; also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=("data_config",))
        self.data_config = data_config
        # self.dataloader_mode = dataloader_mode
        # self.use_test_samples_from = use_test_samples_from

        # attributes to be defined elsewhere
        self.precip_dataset: Optional[Dataset] = None
        self.train_sampler: Optional[SubsetRandomSampler] = None
        self.val_sampler: Optional[SubsetRandomSampler] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

        return

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        if self.data_config.uniform_dequantization:
            raise NotImplementedError('Uniform dequantization not yet supported.')

        if self.hparams.dataloader_mode in ['train', 'eval_set_random']:
            self.precip_dataset = DailyAggregateRainfallDataset(self.data_config)
            dataset_size = len(self.precip_dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(self.data_config.train_val_split * dataset_size))
            train_indices, val_indices = indices[split:], indices[:split]
            generator = torch.Generator().manual_seed(self.hparams.seed)
            self.train_sampler = SubsetRandomSampler(train_indices, generator=generator)
            self.val_sampler = SubsetRandomSampler(val_indices, generator=generator)
        elif self.hparams.dataloader_mode == 'specify_eval':
            self.precip_dataset = RainfallSpecifiedInference(self.data_config, self.hparams.specify_eval_targets)
        elif self.hparams.dataloader_mode == 'eval_set_deterministic':
            self.precip_dataset = PreSavedPrecipDataset(self.data_config, self.hparams.use_test_samples_from,
                                                        self.hparams.stop_at_batch)
        return

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        self.train_loader = DataLoader(self.precip_dataset,
                                       batch_size=self.hparams.batch_size,
                                       num_workers=self.hparams.num_workers,
                                       sampler=self.train_sampler,
                                       pin_memory=self.hparams.pin_memory,
                                       timeout=3600)
        return self.train_loader

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        self.val_loader = DataLoader(self.precip_dataset,
                                     batch_size=self.hparams.batch_size,
                                     timeout=3600,  # 120,
                                     num_workers=self.hparams.num_workers,  # TODO: specify num_workers for val
                                     sampler=self.val_sampler)
        return self.val_loader

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        assert self.hparams.dataloader_mode in ['specify_eval', 'eval_set_random', 'eval_set_deterministic']
        if self.hparams.dataloader_mode == 'specify_eval':
            self.test_loader = DataLoader(self.precip_dataset,
                                          batch_size=1,
                                          timeout=0,
                                          num_workers=1,
                                          collate_fn=xarray_collate_fn)
        elif self.hparams.dataloader_mode == 'eval_set_random':
            raise NotImplementedError()
        else:
            self.test_loader = DataLoader(self.precip_dataset,
                                          batch_size=1,  # hard coded for now
                                          timeout=0,
                                          num_workers=1,
                                          collate_fn=do_nothing_collate_fn)

        return self.test_loader

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    # debug
    with initialize(version_base=None, config_path="../../configs/data", job_name="evaluation"):
        config = compose(config_name="cpc_mrms_data")
    data_module = PrecipDataModule(config.data)
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    print(f"Train DataLoader initialized with {len(train_loader)} batches.")
    # Get the first batch
    first_batch = next(iter(train_loader))
    print(f"First batch: {first_batch}")
