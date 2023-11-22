from pathlib import Path
from collections import defaultdict
from neptune.types import File

import numpy as np
import pandas as pd
import torch
from lightning import LightningModule

from torchmetrics import MeanMetric
from torchmetrics.metric import Metric

from src.mri_utils.io import save_reconstructions


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModule(LightningModule):
    """
    Abstract super class of a `LightningModule` for MRI Reconstrution.
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        num_log_images: int = 8,
    ) -> None:
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=['net'])

        self.num_log_images = num_log_images

        # metric objects for calculating and averaging accuracy across batches
        self.val_log_indices = None
        self.test_log_indices = None
        self.train_log_indices = [0]
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Metrics
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()
        self.train_loss = DistributedMetricSum()
        
    def log_image(self, name, image):
        self.logger.experiment.log_image(image.cpu(), name)
        
    def log_image_neptune(self, name, image, key):
        self.logger.experiment[name].log(File.as_image(image.cpu()), name=key)
        
    def on_validation_epoch_end(self) -> None:
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        
        for val_log in self.validation_step_outputs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = metrics["psnr"] + 20 * torch.log10(torch.tensor(max_vals[fname], dtype=mse_val.dtype, device=mse_val.device)) - 10 * torch.log10(mse_val)
            metrics["ssim"] = metrics["ssim"] + torch.mean(torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()]))


        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("val/loss", val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)
            
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        outputs = defaultdict(dict)
        
        df = pd.DataFrame(columns=['nmse', 'ssim', 'psnr'])
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for val_log in self.test_step_outputs:
            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"].keys():
                max_vals[k] = val_log["max_vals"][k]

        nmse_list = []
        ssim_list = []
        psnr_list = []
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])            )
            target_norm = torch.mean(torch.cat([v.view(-1) for _, v in target_norms[fname].items()])            )
            nmse_val = (mse_val / target_norm).cpu().numpy()
            psnr_val = (20 * torch.log10(torch.tensor(max_vals[fname], dtype=mse_val.dtype, device=mse_val.device)) - 10 * torch.log10(mse_val)).cpu().numpy()
            ssim_val = torch.mean(torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])).cpu().numpy()
            nmse_list.append(nmse_val)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            df.loc[fname] = [nmse_val, psnr_val, ssim_val]
        
        for log in self.test_step_outputs:
            for i, (fname, slice_num) in enumerate(zip(log['fname'], log['slice_num'])):
                outputs[fname][slice_num] = log['output'][i].cpu()
                
        # stack all the slices for each file
        for fname in outputs:
            outputs[fname] = np.stack([out for _, out in sorted(outputs[fname].items())])
            
        #use the default_root_dir if we have a trainer, otherwise save to cwd
        dir_name = "retain_forget_dataset"
        if hasattr(self, "trainer"):
            save_dir = Path(self.trainer.default_root_dir) / dir_name
        else:
            save_dir = Path.cwd()/ dir_name
            
        self.print(f"Savings reconstructions to {save_dir}")
        
        save_reconstructions(outputs, save_dir)
        df.to_csv(str(save_dir) + '/' + dir_name+'.csv', mode='a', header=False)

        
if __name__ == "__main__":
    _ = MriModule(None)
