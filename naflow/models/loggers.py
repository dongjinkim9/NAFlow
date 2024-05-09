from PIL import Image
import os
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pathlib import Path

from typing import Any, Dict, Optional, Union
from lightning.fabric.utilities.types import _PATH

__all__ = [
    "TensorBoardLogger",
    "WandbLogger",
]


class LocalImageLogger(Logger):
    def __init__(
        self,         
        save_dir: _PATH,
        name: Optional[str] = "lightning_logs",
        version: Optional[Union[int, str]] = None,
    ):
        super().__init__()
        self._root_dir = save_dir
        self._name = name
        self._version = version
    
    @property
    def name(self) -> str:
        """Get the name of the experiment.

        Returns:
            The name of the experiment.

        """
        return self._name

    @property
    def version(self) -> Union[int, str]:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.

        """
        if self._version is None:
            self._version = 'temp'
        return self._version

    @property
    def root_dir(self) -> str:
        """Gets the save directory where the TensorBoard experiments are saved.

        Returns:
            The local path to the save directory where the TensorBoard experiments are saved.

        """
        return self._root_dir
    
    @property
    def log_dir(self) -> str:
        """The directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, self.name, version)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir
    
    @property
    @rank_zero_experiment
    def experiment(self) -> "self":
        """Actual object. To use features anywhere in your code, do the following.

        Example::

            logger.experiment.some_function()

        """
        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"
        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        return self

    def log_image(self, name, image, step=None):
        # Convert tensor to PIL Image and save
        image = Image.fromarray((image*255).permute(1, 2, 0).byte().cpu().numpy())
        image.save(Path(self.log_dir) / fr"{name}_{step}.png")

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass