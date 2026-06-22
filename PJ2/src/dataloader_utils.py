"""Fast DataLoader helpers: case-grouped batches for fewer npz reads."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator

import torch
from torch.utils.data import BatchSampler, DataLoader

from src.dataset import DenoiseSliceDataset


class CaseGroupedBatchSampler(BatchSampler):
    """Batch slices from the same case together to maximize volume cache hits."""

    def __init__(self, dataset: DenoiseSliceDataset, batch_size: int, *, shuffle: bool = True) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        case_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, (caseid, _slice_idx) in enumerate(dataset.index):
            case_to_indices[caseid].append(idx)
        self._case_to_indices = dict(case_to_indices)

    def __iter__(self) -> Iterator[list[int]]:
        cases = list(self._case_to_indices.keys())
        if self.shuffle:
            random.shuffle(cases)
        for caseid in cases:
            indices = list(self._case_to_indices[caseid])
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        total = 0
        for indices in self._case_to_indices.values():
            n = len(indices)
            total += (n + self.batch_size - 1) // self.batch_size
        return total


def make_slice_dataloader(
    dataset: DenoiseSliceDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    case_grouped: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    loader_kwargs: dict = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    if case_grouped:
        batch_sampler = CaseGroupedBatchSampler(dataset, batch_size, shuffle=shuffle)
        return DataLoader(dataset, batch_sampler=batch_sampler, **loader_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        **loader_kwargs,
    )
