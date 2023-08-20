# -*- coding: utf-8 -*-

import numpy as np
import os
import pytorch_lightning as pl
import torch
import webdataset as wds
from torchvision.transforms import transforms

from configs.utils import instantiate_from_config


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


class WebDataModuleFromConfig_Laion_Lexica(pl.LightningDataModule):

    def __init__(self,
                 tar_base1,
                 tar_base2,
                 batch_size,
                 train=None,
                 validation=None,
                 test=None,
                 num_workers=4,
                 multinode=True,
                 min_size=None,
                 max_pwatermark=1.0,
                 **kwargs):
        super().__init__()
        print(f'Setting tar base to {tar_base1} and {tar_base2}')
        self.tar_base1 = tar_base1
        self.tar_base2 = tar_base2
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.multinode = multinode
        self.min_size = min_size  # filter out very small images
        self.max_pwatermark = max_pwatermark  # filter out watermarked images

    def make_loader(self, dataset_config):
        image_transforms = [instantiate_from_config(tt) for tt in dataset_config.image_transforms]
        image_transforms = transforms.Compose(image_transforms)

        process = instantiate_from_config(dataset_config['process'])

        shuffle = dataset_config.get('shuffle', 0)
        shardshuffle = shuffle > 0

        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only

        # make dataset for laion
        tars_1 = os.path.join(self.tar_base1, dataset_config.shards1)

        dset1 = wds.WebDataset(
            tars_1, nodesplitter=nodesplitter, shardshuffle=shardshuffle,
            handler=wds.warn_and_continue).repeat().shuffle(shuffle)
        print(f'Loading webdataset with {len(dset1.pipeline[0].urls)} shards.')

        dset1 = (
            dset1.select(self.filter_keys).decode('pil',
                                                  handler=wds.warn_and_continue).select(self.filter_size).map_dict(
                                                      jpg=image_transforms, handler=wds.warn_and_continue).map(process))
        dset1 = (dset1.batched(self.batch_size, partial=False, collation_fn=dict_collation_fn))

        # make dataset for lexica
        tars_2 = os.path.join(self.tar_base2, dataset_config.shards2)

        dset2 = wds.WebDataset(
            tars_2, nodesplitter=nodesplitter, shardshuffle=shardshuffle,
            handler=wds.warn_and_continue).repeat().shuffle(shuffle)

        dset2 = (
            dset2.decode('pil',
                         handler=wds.warn_and_continue).map_dict(jpg=image_transforms,
                                                                 handler=wds.warn_and_continue).map(process))
        dset2 = (dset2.batched(self.batch_size, partial=False, collation_fn=dict_collation_fn))

        # get the corresponding prob
        shards1_prob = dataset_config.get('shards1_prob', 0)
        shards2_prob = dataset_config.get('shards2_prob', 0)
        dataset = wds.RandomMix([dset1, dset2], [shards1_prob, shards2_prob])

        loader = wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=self.num_workers)

        return loader

    def filter_size(self, x):
        if self.min_size is None:
            return True
        try:
            return x['json']['original_width'] >= self.min_size and x['json']['original_height'] >= self.min_size and x[
                'json']['pwatermark'] <= self.max_pwatermark
        except Exception:
            return False

    def filter_keys(self, x):
        try:
            return ("jpg" in x) and ("txt" in x)
        except Exception:
            return False

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None