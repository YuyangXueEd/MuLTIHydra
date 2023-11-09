from typing import Tuple
from collections import defaultdict
import pandas as pd

import torch
import numpy as np
from lightning import LightningModule
from piqa import SSIM
from torchmetrics import MeanMetric
from torchmetrics.image import (
    PeakSignalNoiseRatio, 
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure
)

from src.models.mri_module import MriModule
from src.models.components.unet import Unet
from src.mri_utils.math import complex_abs

class UnetModule(MriModule):
    
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        **kwargs
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=True, ignore=['net'])
        self.net = net
        
        self.l1loss = torch.nn.L1Loss()
        self.ssimloss = SSIM(n_channels=1).cuda()
        
    def forward(self, image):
        if image.ndim == 3:
            return self.net(image.unsqueeze(1)).squeeze(1)
        else:
            return self.net(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.PSNR.reset()
        self.SSIM.reset()
        self.MSSSIM.reset()
        
    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
        """
        output = self.forward(batch.image)
        # print(output.shape)
        loss = self.l1loss(output, batch.target)# + (1 - self.ssimloss(output.unsqueeze(1), batch.target.unsqueeze(1)))
        return loss, output
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the batch.

        :return: A tensor of losses.
        """
        loss, _ = self.model_step(batch)
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
        
        loss, output = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, prog_bar=True)

        if output.ndim == 4:
            mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        elif output.ndim == 3:
            mean = batch.mean.unsqueeze(1).unsqueeze(2)
            std = batch.std.unsqueeze(1).unsqueeze(2)
            
        if output.ndim == 2:
            output = output.unsqueeze(0)
        elif output.ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        
        if batch.target.ndim == 2:
            batch.target = batch.target.unsqueeze(0)
        elif batch.target.ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        if self.val_log_indices is None:
            self.val_log_indices = list(
            np.random.permutation(len(self.trainer.val_dataloaders))[: self.num_log_images]
        )
            
        if isinstance(batch_idx, int):
            batch_indices = [batch_idx]
        else:
            batch_indices = batch_idx
            
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_image_{batch_idx}"
                image = (batch.image * std + mean).unsqueeze(1)
                target = (batch.target * std + mean).unsqueeze(1)
                output_vis = (output * std + mean).unsqueeze(1)
                error = torch.abs(target - output_vis)
                image = image / image.max()
                output_vis = output_vis / output_vis.max()
                target = target / target.max()
                error = error / error.max()
                
                # for comet
                self.log_image(f"{key}/image", image[i])
                self.log_image(f"{key}/target", target[i])
                self.log_image(f"{key}/reconstruction", output_vis[i])
                self.log_image(f"{key}/error", error[i])
                # for neptune
                # self.log_image_neptune("image", image[0], key)
                # self.log_image_neptune("target", target[0], key)
                # self.log_image_neptune("recon", output_vis[0], key)
                # self.log_image_neptune("error", error[0], key)
                
        pred = {
            "loss": loss,
            "psnr": self.PSNR(output, batch.target),
            "ssim": self.SSIM(output[None, ...], batch.target[None, ...]),
            "msssim": self.MSSSIM(output[None, ...], batch.target[None, ...])
        }
        
        self.validation_step_outputs.append(pred)
        
        return pred

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = np.array([])
        psnr = np.array([])
        ssim = np.array([])
        msssim = np.array([])
        for results_dict in self.validation_step_outputs:
            loss = np.append(loss, results_dict["loss"].cpu().numpy())
            psnr = np.append(psnr, results_dict["psnr"].cpu().numpy())
            ssim = np.append(ssim, results_dict["ssim"].cpu().numpy())
            msssim = np.append(msssim, results_dict["msssim"].cpu().numpy())

        self.log("val/loss", loss.mean(), sync_dist=True)
        self.log("val/psnr", psnr.mean(), sync_dist=True)
        self.log("val/ssim", ssim.mean(), sync_dist=True)
        self.log("val/msssim", msssim.mean(), sync_dist=True)
        
        # self.logger.experiment["val/loss"] = loss.mean()
        # self.logger.experiment["val/psnr"] = psnr.mean()
        # self.logger.experiment["val/ssim"] = ssim.mean()
        # self.logger.experiment["val/msssim"] = msssim.mean()
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, output = self.model_step(batch)
        # update and log metrics
        # self.test_loss(loss)
        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        if output.ndim == 4:
            mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        elif output.ndim == 3:
            mean = batch.mean.unsqueeze(1).unsqueeze(2)
            std = batch.std.unsqueeze(1).unsqueeze(2)
            
        if output.ndim == 2:
            output = output.unsqueeze(0)
        elif output.ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        
        if batch.target.ndim == 2:
            batch.target = batch.target.unsqueeze(0)
        elif batch.target.ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
            
        pred = {
            "loss": loss,
            "psnr": self.PSNR(output, batch.target),
            "ssim": self.SSIM(output[None, ...], batch.target[None, ...]),
            "msssim": self.MSSSIM(output[None, ...], batch.target[None, ...])
        }
        
        self.test_step_outputs.append(pred)
        
        return pred

    def on_test_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        
        df = pd.DataFrame(columns=['nmse', 'ssim', 'psnr'])
        
        loss = np.array([])
        psnr = np.array([])
        ssim = np.array([])
        msssim = np.array([])
        for results_dict in self.test_step_outputs:
            loss = np.append(loss, results_dict["loss"].cpu().numpy())
            psnr = np.append(psnr, results_dict["psnr"].cpu().numpy())
            ssim = np.append(ssim, results_dict["ssim"].cpu().numpy())
            msssim = np.append(msssim, results_dict["msssim"].cpu().numpy())

        self.log("test/loss", loss.mean(), sync_dist=True)
        self.log("test/psnr", psnr.mean(), sync_dist=True)
        self.log("test/ssim", ssim.mean(), sync_dist=True)
        self.log("test/msssim", msssim.mean(), sync_dist=True)
        
        # self.logger.experiment["test/loss"] = loss.mean()
        # self.logger.experiment["test/psnr"] = psnr.mean()
        # self.logger.experiment["test/ssim"] = ssim.mean()
        # self.logger.experiment["test/msssim"] = msssim.mean()
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
    _ = UnetModule()
