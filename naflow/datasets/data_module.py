from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from utils.common import instantiate_from_config

class RepeatDataset():
    def __init__(self, dataset, times, iterations=None, batch_size=-1):
        self.dataset = dataset
        self.times = times
        self.iterations = iterations
        self.batch_size = batch_size 
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        if self.iterations is None:
            return self.times * self._ori_len
        else:
            return self.iterations * self.batch_size

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, train_config: str, val_config: str=None):
        super().__init__()
        self.train_config = OmegaConf.load(train_config)
        self.val_config = OmegaConf.load(val_config) if val_config else None
            
    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = instantiate_from_config(self.train_config["dataset"])
            
            repeat_rate = self.train_config["repeat_dataset"]["times"]
            if repeat_rate > 1:
                self.train_dataset = RepeatDataset(self.train_dataset, repeat_rate)

            if self.val_config:
                self.val_dataset = instantiate_from_config(self.val_config["dataset"])
            else:
                self.val_dataset = None
        elif stage == "validate":
            self.val_dataset = instantiate_from_config(self.val_config["dataset"])
        else:
            raise NotImplementedError(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset, **self.train_config["data_loader"]
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_dataset is None:
            return None
        return DataLoader(
            dataset=self.val_dataset, **self.val_config["data_loader"]
        )