"""
Extension to pytorch batch sampler to also yield a random scalar between a given range.
"""

import random

from torch.utils.data import Sampler
from torch.utils.data import SequentialSampler
from torch._six import int_classes as _int_classes

class BatchSamplerRandScale(Sampler):
    r"""Extending the Batch Sampler to also pass a scale factor for
        random scale between a list of ranges.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
            with ``__len__`` implemented.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        scale_range (List): The range in which will be the sample will be randomly scaled

    Example:
        >>> list(BatchSamplerRandScale(SequentialSampler(range(10)),
                batch_size=3, drop_last=False, scale_range=[0.5,1]))
        [[(0, 0.65), (1, 0.65), (2, 0.65)],
         [(3, 0.8), (4, 0.8), (5, 0.8)],
         [(6, 0.93), (7, 0.93), (8, 0.93)],
         [(9, 0.54)]]
    """

    def __init__(self, sampler, batch_size, drop_last, scale_range):
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        super().__init__(None)
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        assert len(scale_range) == 2
        self.scale_range = scale_range

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                scale_factor = random.uniform(*self.scale_range)
                batch = [(x, scale_factor) for x in batch]
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            scale_factor = random.uniform(*self.scale_range)
            batch = [(x, scale_factor) for x in batch]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

if __name__ == "__main__":
    test = list(BatchSamplerRandScale(SequentialSampler(range(10)),
        batch_size=3, drop_last=False, scale_range=[0.5, 1]))
    print(test)
