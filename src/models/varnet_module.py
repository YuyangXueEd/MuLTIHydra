from typing import Tuple
from collections import defaultdict
import pandas as pd

import torch
import numpy as np

from src.data.components.fastmri_transform_utils import center_crop_to_smallest, center_crop
from src.models.mri_module import MriModule
from src.models.losses.ssim import SSIMLoss
from src.utils.evaluate import mse, ssim



class VarNetModule(MriModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        **kwargs
    ):
        super().__init__( **kwargs)
        self.save_hyperparameters(logger=True, ignore=['net'])
        self.net = net
        
        self.l1loss = torch.nn.L1Loss()
        self.ssimloss = SSIMLoss()
        
    def forward(self, masked_kspace, mask, num_low_frequencies):
        return self.net(masked_kspace, mask, num_low_frequencies)
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        
		self.NMSE.reset()
        self.SSIM.reset()
        self.PSNR.reset()
        self.ValLoss.reset()
        self.TotExamples.reset()
        self.TotSliceExamples.reset()
        self.train_loss.reset()
    
    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
        """
        output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies)

        target, output = center_crop_to_smallest(batch.target, output)
        loss = self.ssimloss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        ) + 1e-5 * self.l1loss(output.unsqueeze(1), target.unsqueeze(1))
        
        return loss, output, target
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the batch.

        :return: A tensor of losses.
        """
        loss, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, prog_bar=False)
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        
        loss, output, target = self.model_step(batch)
        # update and log metrics
        

        if output.ndim == 2:
            output = output.unsqueeze(0)
        elif output.ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if target.ndim == 2:
            target = target.unsqueeze(0)
        elif target.ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        
        
        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders))[
                    : self.num_log_images
                ]
            )
            
        # log images to logger
        if isinstance(batch_idx, int):
            batch_indices = [batch_idx]
        else:
            batch_indices = batch_idx
            
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_image_idx_{batch_idx}"
                target_vis = target[i].unsqueeze(0)
                output_vis = output[i].unsqueeze(0)
                error_vis = torch.abs(target_vis - output_vis)
                output_vis = output_vis / output_vis.max()
                target_vis = target_vis / target_vis.max()
                error_vis = error_vis / error_vis.max()
                
                # for comet
                self.log_image(f"{key}/target", target_vis)
                self.log_image(f"{key}/reconstruction", output_vis)
                self.log_image(f"{key}/error", error_vis)
                # for neptune
                # self.log_image_neptune("image", image[0], key)
                # self.log_image_neptune("target", target[0], key)
                # self.log_image_neptune("recon", output_vis[0], key)
                # self.log_image_neptune("error", error[0], key)
        
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        
        for i, fname in enumerate(batch.fname):
            slice_num = int(batch.slice_num[i].cpu())
            maxval = batch.max_value[i].cpu().numpy()
            output_i = output[i].cpu().numpy()
            target_i = target[i].cpu().numpy()
            
            mse_vals[fname][slice_num] = torch.tensor(mse(target_i, output_i)).view(1)
            target_norms[fname][slice_num] = torch.tensor(mse(target_i, np.zeros_like(target_i))).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(ssim(target_i[None, ...], output_i[None, ...], maxval=maxval)).view(1)
            max_vals[fname] = maxval
        
        pred = {
            "val_loss": loss,
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "max_vals": max_vals
        }
        
        self.validation_step_outputs.append(pred)
        
        return pred
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        
        loss, output, target = self.model_step(batch)
        # check for FLAIR 203
        if output.shape[-1] < batch.crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        else:
            crop_size = batch.crop_size

        output = center_crop(output, crop_size)

        pred = {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }
        self.test_step_outputs.append(pred)
        
        return pred
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    _ = VarnetModule()
