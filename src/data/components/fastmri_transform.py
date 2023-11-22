from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union
from pathlib import Path
import numpy as np
import torch

from src.data.components.fastmri_transform_utils import *
from src.mri_utils.math import complex_abs
from src.mri_utils.coil_combine import rss, rss_complex
from src.mri_utils.fftc import fft2c_new as fft2c, ifft2c_new as ifft2c

class UnetSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        image: Subsampled image after inverse FFT.
        target: The target image (if applicable).
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
    """

    image: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    
    
class UnetDataTransform:
    """
    Data Transformer for U-Net like models training.
    """
    def __init__(
        self,
        challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """

        Args:
            challenge (str): Challenge from ("singlecoil", "multicoil").
            mask_func (Callable[MaskFunc], optional): 
                A function that can create a mask of appropriate shape.
            use_seed (bool, optional): 
                If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError(f"Challenge should be either 'singlecoil' or 'multicoil', got {challenge}.")
        
        self.challenge = challenge
        self.mask_func = mask_func
        self.use_seed = use_seed
        
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int
    ) -> UnetSample:
        """

        Args:
            kspace (np.ndarray): 
                Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask (np.ndarray): Mask from the test dataset.
            target (np.ndarray): Target image.
            attrs (Dict): Acquisition related information.
            fname (str): File name.
            slice_num (int): Serial number of the slice.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]: _description_
        """
        
        kspace_torch = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace = apply_mask(kspace_torch, self.mask_func, seed=seed)[0]
        else:
            masked_kspace = kspace_torch
        # inverse Fourier transform to get zero filled solution
        image = ifft2c(masked_kspace)
        crop_size=(attrs["recon_size"][0], attrs["recon_size"][1])
        image = complex_center_crop(image, crop_size)
        
        # complex_abs
        if self.challenge == 'multicoil':
            image = rss_complex(image)
        else:
            image = complex_abs(image)
            
        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target_torch = to_tensor(target)
            target_torch = center_crop(target_torch, crop_size)
            target_torch = normalize(target_torch, mean, std, eps=1e-11)
            target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = torch.Tensor([0])
            
        return UnetSample(
            image=image,
            target=target_torch,
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
        )
        

class VarNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.


    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """
    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]
    
class VarNetDataTransform:
    """
    Data Transformer for variational network training.
    """
    def __init__(
        self,
        mask_func: Optional[MaskFunc] = None,
        challenge: str = "multicoil",
        use_seed: bool = True,
    ):
        """

        Args:
            mask_func (Callable[MaskFunc], optional): 
                A function that can create a mask of appropriate shape.
            challenge (str, optional): 
                Challenge from ("singlecoil", "multicoil").
            use_seed (bool, optional): 
                If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
            num_low_frequencies (int, optional): 
                Number of low-frequency samples to keep in the center of
                k-space. If None, no low-frequency samples are kept.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError(f"Challenge should be either 'singlecoil' or 'multicoil', got {challenge}.")
        
        self.mask_func = mask_func
        self.challenge = challenge
        self.use_seed = use_seed
        
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int) -> VarNetSample:
        """

        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """
        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        else:
            target_torch = torch.Tensor([0])
            max_value = 0.0
        
        kspace_torch = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]
        
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])
        
        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )
            
            sample = VarNetSample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=num_low_frequencies,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )
        else:
            masked_kspace = kspace_torch
            shape = np.array(kspace_torch.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask_torch = mask_torch.reshape(*mask_shape)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0

            sample = VarNetSample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=0,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )

        return sample
            