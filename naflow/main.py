from argparse import ArgumentParser

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pathlib import Path
import torch
import wandb

from utils.common import instantiate_from_config, load_state_dict
from pytorch_lightning.loggers import WandbLogger

import torch

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    if config.lightning.seed:
        pl.seed_everything(config.lightning.seed, workers=True)
    
    data_module = instantiate_from_config(config.data)
    model = instantiate_from_config(OmegaConf.load(config.model.config))

    if config.model.get("resume"):
        load_state_dict(model, torch.load(config.model.resume, map_location="cpu"), strict=True)

    callbacks = []
    if 'callbacks' in config.lightning.keys():
        for callback_config in config.lightning.callbacks:
            callbacks.append(instantiate_from_config(callback_config))

    if not args.debug:
        loggers = []
        if 'loggers' in config.lightning.keys():
            for logger_config in config.lightning.loggers:
                logger = instantiate_from_config(logger_config)
                loggers.append(logger)
                if isinstance(logger, WandbLogger):
                    code = wandb.Artifact('code', type='code')
                    for path in Path('.').glob('**/*.py'):
                        code.add_file(path, name=str(path))
                    for path in Path('.').glob('**/*.yaml'):
                        code.add_file(path, name=str(path))
                    logger.experiment.log_artifact(code)

    if args.debug:
        loggers, callbacks = [], []
        debug_logger = instantiate_from_config(
            {'target': 'models.loggers.LocalImageLogger',
             'params': {
                 'save_dir' : './logs/',
                 'name': 'LocalImageLogger',
                 'version': 'debug',
             }}
        )
        loggers.append(debug_logger)
        config.lightning.trainer.val_check_interval = 100
        data_module.train_config.dataset.params.preload = False
        data_module.val_config.dataset.params.preload = False

    trainer = pl.Trainer(callbacks=callbacks, logger=loggers, **config.lightning.trainer)
    if config.lightning.mode == 'fit':
        trainer.fit(model, datamodule=data_module)
    elif config.lightning.mode == 'validate':
        trainer.validate(model, datamodule=data_module)
    else:
        assert False, f'unsupported mode : {config.lightning.mode}'


if __name__ == "__main__":
    main()
