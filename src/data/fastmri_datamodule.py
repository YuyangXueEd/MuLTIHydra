import os
import random
import h5py
import xml.etree.ElementTree as etree
from typing import Any, Dict, Optional, Tuple, Callable, Union, NamedTuple, List
import pickle
from pathlib import Path
import torch
import numpy as np
from lightning import LightningDataModule
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import transforms

from src.data.components.fastmri_transform_utils import et_query
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: SliceDataset = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if data.transform.mask_func is not None:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))


def _check_both_not_none(val1, val2):
    if (val1 is not None) and (val2 is not None):
        return True

    return False


class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]


class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to FastMRI slices.
    """
    
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None
    ):
        """

        Args:
            root (Union[str, Path, os.PathLike]): Path to the dataset
            challenge (str): "singlecoil" or "multicoil"
            transform (Optional[Callable], optional): 
                Optional; A callable object that pre-processes the raw data
                into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data
            use_dataset_cache (bool, optional): 
                Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate (Optional[float], optional): 
                Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate (Optional[float], optional): 
                Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file (Union[str, Path, os.PathLike], optional): 
                Optional; A file in which to cache dataset information for faster load times. 
                Defaults to "dataset_cache.pkl".
            num_cols (Optional[Tuple[int]], optional): 
                Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter (Optional[Callable], optional):
                Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.


        Returns:
            torch.Tensor: Data sample 
        """
        
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError(f'Challenge should be either "singlecoil" or "multicoil"')
        
        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError("Either sample_rate or volume_sample_rate should be set but not both.")
        
        self.dataset_cache_file = Path('../data/caches/' + dataset_cache_file)
        self.transform = transform
        self.recons_key = "reconstruction_rss" if challenge == "multicoil" else "reconstruction_esc"
        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}
                
        # check if our dataset is in the cache
        # if there use that metadata, if not then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)
                
                new_raw_samples = []
                for slice_ind in range(num_slices):
                    raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                    if self.raw_sample_filter(raw_sample):
                        new_raw_samples.append(raw_sample)
                        
                self.raw_samples += new_raw_samples
                
            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                log.info(f"Saving dataset cache to {self.dataset_cache_file}")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
                    
        else:
            log.info(f"Using dataset cache from {self.dataset_cache_file}")
            self.raw_samples = dataset_cache[root]
            
        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0
            
        if sample_rate < 1.0: # sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.raw_samples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.raw_samples = [
                raw_sample
                for raw_sample in self.raw_samples
                if raw_sample[0].stem in sampled_vols
            ]
            
        if num_cols:
            self.raw_samples = [
                ex for ex in self.raw_samples 
                if ex[2]["encoding_size"][1] in num_cols
            ]
                
    @staticmethod
    def _retrieve_metadata(fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices
    
    def __len__(self):
        return len(self.raw_samples)
    
    def __getitem__(self, i:int):
        fname, dataslice, metadata = self.raw_samples[i]
        
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None
            
            attrs = dict(hf.attrs)
            attrs.update(metadata)
            
        if self.transform:
            sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice)
        else:
            sample = (kspace, mask, target, attrs, fname.name, dataslice)
            
        return sample


class VolumeSampler(Sampler):
    """
    Sampler for volumetric MRI data.

    Based on pytorch DistributedSampler, the difference is that all instances
    from the same MRI volume need to go to the same node for distributed
    training. Dataset example is a list of tuples (fname, instance), where
    fname is essentially the volume name (actually a filename).
    """

    def __init__(
        self,
        dataset: SliceDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            dataset: An MRI dataset (e.g., SliceData).
            num_replicas: Number of processes participating in distributed
                training. By default, :attr:`rank` is retrieved from the
                current distributed group.
            rank: Rank of the current process within :attr:`num_replicas`. By
                default, :attr:`rank` is retrieved from the current distributed
                group.
            shuffle: If ``True`` (default), sampler will shuffle the indices.
            seed: random seed used to shuffle the sampler if
                :attr:`shuffle=True`. This number should be identical across
                all processes in the distributed group.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        # get all file names and split them based on number of processes
        self.all_volume_names = sorted(
            set(str(raw_sample[0]) for raw_sample in self.dataset.raw_samples)
        )
        self.all_volumes_split: List[List[str]] = []
        for rank_num in range(self.num_replicas):
            self.all_volumes_split.append(
                [
                    self.all_volume_names[i]
                    for i in range(
                        rank_num, len(self.all_volume_names), self.num_replicas
                    )
                ]
            )

        # get slice indices for each file name
        rank_indices: List[List[int]] = [[] for _ in range(self.num_replicas)]
        for i, raw_sample in enumerate(self.dataset.raw_samples):
            vname = str(raw_sample[0])
            for rank_num in range(self.num_replicas):
                if vname in self.all_volumes_split[rank_num]:
                    rank_indices[rank_num].append(i)
                    break

        # need to send equal number of samples to each process - take the max
        self.num_samples = max(len(indices) for indices in rank_indices)
        self.total_size = self.num_samples * self.num_replicas
        self.indices = rank_indices[self.rank]

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ordering = torch.randperm(len(self.indices), generator=g).tolist()
            indices = [self.indices[i] for i in ordering]
        else:
            indices = self.indices

        # add extra samples to match num_samples
        repeat_times = self.num_samples // len(indices)
        indices = indices * repeat_times
        indices = indices + indices[: self.num_samples - len(indices)]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
            

class FastMRIDataModule(LightningDataModule):
    """
    `LightningDataModule` for the FastMRI dataset.
    
    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.
    
    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.
    
    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.

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

        # def predict_dataloader(self):
        # return predict dataloader

        # def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```
    """
    
    def __init__(
        self,
        data_path: str,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        #combine_train_val: bool = False,
        test_split: str="test",
        test_path: Optional[str] = None,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        test_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        test_volume_sample_rate: Optional[float] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        test_filter: Optional[Callable] = None,
        use_dataset_cache_file: bool = False,
        dataset_cache_file: str = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
    ):
        """_summary_

        Args:
            data_path (str): Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            challenge (str): Name of challenge from ('multicoil', 'singlecoil').
            train_transform (Callable): A transform object for the training split.
            val_transform (Callable): A transform object for the validation split.
            test_transform (Callable): A transform object for the test split.
            test_split (str, optional): Name of test split from ("test", "challenge").
            test_path (Optional[str], optional): 
                An optional test path. Passing this overwrites data_path and test_split.
            sample_rate (Optional[float], optional): 
                Fraction of slices of the training data split to use.
                Can be set to less than 1.0 for rapid prototyping. If not set,
                it defaults to 1.0. To subsample the dataset either set
                sample_rate (sample by slice) or volume_sample_rate (sample by
                volume), but not both.
            val_sample_rate (Optional[float], optional): Same as sample_rate, but for val split.
            test_sample_rate (Optional[float], optional): Same as sample_rate, but for test split.
            volume_sample_rate (Optional[float], optional): 
                Fraction of volumes of the training data split
                to use. Can be set to less than 1.0 for rapid prototyping. If
                not set, it defaults to 1.0. To subsample the dataset either
                set sample_rate (sample by slice) or volume_sample_rate (sample
                by volume), but not both.
            val_volume_sample_rate (Optional[float], optional): 
                Same as volume_sample_rate, but for val split.
            test_volume_sample_rate (Optional[float], optional):
                Same as volume_sample_rate, but for test split.
            train_filter (Optional[Callable], optional): 
                A callable which takes as input a training example metadata, 
                and returns whether it should be part of the training dataset.
            val_filter (Optional[Callable], optional): Same as train_filter, but for val split.
            test_filter (Optional[Callable], optional): Same as train_filter, but for test split.
            use_dataset_cache_file (bool, optional): 
                Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
            dataset_cache_file (str, optional): 
                The filename of the dataset cache file.
            batch_size (int, optional): Batch size.
            num_workers (int, optional): Number of workers for PyTorch dataloader.
            distributed_sampler (bool, optional): 
                Whether to use a distributed sampler. This should be set to True if training with ddp.
        """
        super().__init__()
        
        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Can set sample_rate or volume_sample_rate, but not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError(
                "Can set val_sample_rate or val_volume_sample_rate, but not both."
            )
        if _check_both_not_none(test_sample_rate, test_volume_sample_rate):
            raise ValueError(
                "Can set test_sample_rate or test_volume_sample_rate, but not both."
            )
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.data_path = Path(data_path)
        self.challenge = challenge
        self.train_transform = train_transform
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        # self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.use_dataset_cache_file = use_dataset_cache_file
        self.dataset_cache_file = dataset_cache_file
        self.num_cols = num_cols
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        
    def _create_data_loader(
            self,
            data_transform: Callable,
            data_partition: str,
            sample_rate: Optional[float] = None,
            volume_sample_rate: Optional[float] = None,
        ) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = (
                self.volume_sample_rate if volume_sample_rate is None else volume_sample_rate
            )
            raw_sample_filter = self.train_filter
        else:
            is_train = False
            if data_partition == "val":
                sample_rate = self.val_sample_rate if sample_rate is None else sample_rate
                volume_sample_rate = (
                    self.val_volume_sample_rate if volume_sample_rate is None else volume_sample_rate
                )
                raw_sample_filter = self.val_filter
                
            elif data_partition == "test":
                sample_rate = self.test_sample_rate if sample_rate is None else sample_rate
                volume_sample_rate = (
                    self.test_volume_sample_rate if volume_sample_rate is None else volume_sample_rate
                )
                raw_sample_filter = self.test_filter
                
            else:
                raise ValueError(f"Invalid data partition {data_partition}")
            
        dataset: SliceDataset
        if data_partition in ("test", "challenge") and self.test_path is not None:
            data_path = Path(self.test_path)
        else:
            data_path = self.data_path / f"{self.challenge}_{data_partition}"
            
        dataset = SliceDataset(
            root=data_path,
            challenge=self.challenge,
            transform=data_transform,
            use_dataset_cache=self.use_dataset_cache_file,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            dataset_cache_file=self.dataset_cache_file,
            num_cols=self.num_cols,
            raw_sample_filter=raw_sample_filter,
        )
        
        sampler = None

        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )

        return dataloader
    
    def prepare_data(self):
        """
        rank 0 ddp process. 
        """
        if self.use_dataset_cache_file:
            if self.test_path is not None:
                test_path = Path(self.test_path)
            else:
                test_path = self.data_path / f"{self.challenge}_test"
            data_paths = [
                self.data_path / f"{self.challenge}_train",
                self.data_path / f"{self.challenge}_val",
                test_path,
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform,
            ]
            for i, (data_path, data_transform) in enumerate(
                zip(data_paths, data_transforms)
            ):
                # NOTE: Fixed so that val and test use correct sample rates
                sample_rate = self.sample_rate  # if i == 0 else 1.0
                volume_sample_rate = self.volume_sample_rate  # if i == 0 else None
                _ = SliceDataset(
                    root=data_path,
                    challenge=self.challenge,
                    transform=data_transform,
                    use_dataset_cache=self.use_dataset_cache_file,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    dataset_cache_file=self.dataset_cache_file,
                    num_cols=self.num_cols,
                    #raw_sample_filter=raw_sample_filter,
                )
                
    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(self.test_transform, data_partition=self.test_split)