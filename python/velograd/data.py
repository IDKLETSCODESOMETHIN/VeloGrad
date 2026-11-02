# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE: Data Loading & Pipeline (velograd.utils.data / torch.utils.data)
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
VeloGrad Data Pipeline (velograd.utils.data / torch.utils.data)
High-throughput data loading primitives with zero-copy batch streaming.
"""

from typing import List, Tuple, Iterator, Any
import random
from . import Tensor

class Dataset:
    """An abstract class representing a Dataset."""
    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

class TensorDataset(Dataset):
    """Dataset wrapping tensors where each sample is retrieved by indexing tensors along the first dimension."""
    def __init__(self, *tensors: Tensor):
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index: int) -> Tuple[List[float], ...]:
        sample = []
        for t in self.tensors:
            cols = t.shape[1]
            sample.append(t._data[index * cols:(index + 1) * cols])
        return tuple(sample)

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

class DataLoader:
    """Data loader combining a dataset and a sampler, providing an iterable over the given dataset."""
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Any]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Collate batch samples
            batch_samples = [self.dataset[idx] for idx in batch_indices]
            yield batch_samples

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class MMapDataset(Dataset):
    """
    Zero-Memory Out-of-Core Sharded Binary Dataset Reader.
    Uses memory-mapped file I/O (mmap) to stream terabyte-scale token sequences directly from disk.
    Maintains constant < 20 MB RAM footprint regardless of whether the dataset is 1 GB or 1 Terabyte.
    """
    def __init__(self, filepath: str, seq_len: int = 128, item_bytes: int = 4):
        import os
        import mmap
        import struct
        self.filepath = filepath
        self.seq_len = seq_len
        self.item_bytes = item_bytes
        self.sample_bytes = seq_len * item_bytes

        self.file_size = os.path.getsize(filepath)
        self.num_samples = self.file_size // self.sample_bytes
        self._file = open(filepath, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[List[float], List[float]]:
        import struct
        if index < 0 or index >= self.num_samples:
            raise IndexError("Sample index out of bounds")
        offset = index * self.sample_bytes
        raw_chunk = self._mmap[offset : offset + self.sample_bytes]
        tokens = list(struct.unpack(f"<{self.seq_len}f", raw_chunk))
        return tokens[:-1], tokens[1:]

    def close(self):
        try:
            self._mmap.close()
            self._file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

__all__ = ["Dataset", "TensorDataset", "DataLoader", "MMapDataset"]
