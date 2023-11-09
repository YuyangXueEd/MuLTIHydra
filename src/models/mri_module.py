from collections import defaultdict
from neptune.types import File

import torch
from lightning import LightningModule

from torchmetrics import MeanMetric
from torchmetrics.image import (
    PeakSignalNoiseRatio, 
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure
)

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
        self.save_hyperparameters(logger=False)

        self.num_log_images = num_log_images

        # metric objects for calculating and averaging accuracy across batches
        self.val_log_indices = None
        self.test_log_indices = None
        self.train_log_indices = [0]
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Metrics
        self.MSSSIM = MultiScaleStructuralSimilarityIndexMeasure()
        self.SSIM = StructuralSimilarityIndexMeasure()
        self.PSNR = PeakSignalNoiseRatio()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        #self.TotExamples = DistributedMetricSum()
        #self.TotSliceExamples = DistributedMetricSum()
        
    def log_image(self, name, image):
        self.logger.experiment.log_image(image.cpu(), name)
        
    def log_image_neptune(self, name, image, key):
        self.logger.experiment[name].log(File.as_image(image.cpu()), name=key)
        
if __name__ == "__main__":
    _ = MriModule(None)
