import torch
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from utils.metrics import calculate_kld, calculate_alkd
from torchvision.utils import make_grid
from utils.common import instantiate_from_config_with_arg
from typing import Mapping, Any
from archs.naflow import NAFlow

class LitNAFlow(LightningModule):
    def __init__(self, 
                 network_G_config: Mapping[str, Any],
                 optimizer_config: Mapping[str, Any],
                 scheduler_config: Mapping[str, Any],):
        super().__init__()

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.model = NAFlow(network_G_config)

        self.add_cond_noise_std = network_G_config['add_cond_noise_std']
        
        self.nan_count, self.skip_count = 0, 0
        self.sampled_images = []

        self.save_hyperparameters()

    def forward(self, noisy_img, clean_img):
        inp, cond = noisy_img, clean_img

        # add minor noise
        if self.add_cond_noise_std > 0.0:
            cond = cond + (torch.randn_like(cond) / 255) * self.add_cond_noise_std
            cond = torch.clamp(cond, 0.0, 1.0)
        
        z, nll, _ = self.model(input=inp.clone(), cond=cond.clone(), reverse=False, label=['S6_00100'])
        resampled_z = self.model.noise_aware_sampling(z)
        output, _ = self.model(cond=cond, z=resampled_z, eps_std=1.0, reverse=True, label=['S6_00100'])

        return output

    def training_step(self, batch, batch_idx):
        lq, gt, label = batch['LQ'], batch['GT'], batch['class']
        self.log("bs_per_gpu", lq.shape[0], prog_bar=True,logger=False,rank_zero_only=True)

        cond = gt.clone()
        # add minor noise
        if self.add_cond_noise_std > 0.0:
            cond = cond + (torch.randn_like(cond) / 255) * self.add_cond_noise_std
            cond = torch.clamp(cond, 0.0, 1.0)

        # Flow loss
        losses = {}
        z, nll, _ = self.model(input=lq, cond=cond, reverse=False, label=label)
        losses['train/nll'] = torch.mean(nll)

        total_loss = sum(losses.values())
        losses['train/total'] = sum(losses.values())

        # skip diverging nll
        if not torch.isfinite(total_loss):
            self.nan_count += 1
            self.log("NaN_count", self.nan_count, prog_bar=True)
            if self.nan_count > 1000:
                self.nan_count = 0
                raise ValueError()
            
            # prevent memory leak
            self.automatic_optimization=False
            self.manual_backward(total_loss)
            self.automatic_optimization=True
            
            return None

        self.log_dict(losses, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        assert batch['GT'].shape[0] == 1
        losses = {}

        # kld, alkd test
        if dataloader_idx == 0:
            gt, lq, label = batch['GT'], batch['LQ'], batch['class']

            cond = gt.clone()
            inp = lq.clone()

            # add minor noise
            if self.add_cond_noise_std > 0.0:
                cond = cond + (torch.randn_like(cond) / 255) * self.add_cond_noise_std
                cond = torch.clamp(cond, 0.0, 1.0)
            
            z, nll, _ = self.model(input=inp.clone(), cond=cond.clone(), reverse=False, label=label)

            # consider the nlls whose classes exist in training phase
            if nll is not None:
                losses['validation/nll_loss'] = torch.mean(nll)
                
            clean, real_noisy = gt.clone(), lq.clone() # tensor [3,H,W]
            
            resampled_z = self.model.noise_aware_sampling(z)
            output, _ = self.model(cond=cond, z=resampled_z, eps_std=1.0, reverse=True, label=label)

            fake_noisy = output.clone()
            fake_noisy = torch.clamp(fake_noisy, 0., 1.0)
            
            real_noise = real_noisy.clone() - clean.clone()
            fake_noise = fake_noisy.clone() - clean.clone()

            # evalutate using noise metrics
            kld = calculate_kld(real_noise[0], fake_noise[0])
            alkd = calculate_alkd(real_noisy, fake_noisy, clean)
            losses['validation/noise-kld'] = kld
            losses['validation/noise-alkd'] = alkd

            # log the outputs!
            self.log_dict(losses, sync_dist=True, prog_bar=True)

            def minmax_norm(x: torch.Tensor):
                return (x - x.min()) / (x.max() - x.min())

            if batch_idx % 100 == 0:
                self.sampled_images.append(clean[0].cpu())
                self.sampled_images.append(real_noisy[0].cpu())
                self.sampled_images.append(minmax_norm(real_noise)[0].cpu())
                self.sampled_images.append(fake_noisy[0].cpu())
                self.sampled_images.append(minmax_norm(fake_noise)[0].cpu())
        

    def log_image(self, key, image):
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(key, image, self.global_step+1)
            if isinstance(logger, WandbLogger):
                logger.experiment.log({key: wandb.Image(image),})

    def on_validation_epoch_end(self):
        grid = make_grid(self.sampled_images, nrow=5)
        self.log_image('validation/sampled_images', grid)
        self.sampled_images.clear() # free memory

    def configure_optimizers(self):
        optimizer = instantiate_from_config_with_arg(
            self.optimizer_config, [{'params': self.model.parameters()}])
        learning_rate_scheduler = instantiate_from_config_with_arg(
            self.scheduler_config, optimizer)
        
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": learning_rate_scheduler,
                    "interval": 'step',
                    "frequency": 1,},}