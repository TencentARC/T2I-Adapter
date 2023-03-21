# -*- coding: utf-8 -*-

import numpy as np
import os
import pytorch_lightning as pl
import torch
import webdataset as wds
from torchvision.transforms import transforms

from ldm.util import instantiate_from_config


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


class WebDataModuleFromConfig(pl.LightningDataModule):

    def __init__(self,
                 tar_base,
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
        print(f'Setting tar base to {tar_base}')
        self.tar_base = tar_base
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

        process_list = []
        for process_config in dataset_config['process']:
            process_list.append(instantiate_from_config(process_config))

        shuffle = dataset_config.get('shuffle', 0)
        shardshuffle = shuffle > 0

        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only

        tars = os.path.join(self.tar_base, dataset_config.shards)

        dset = wds.WebDataset(
            tars, nodesplitter=nodesplitter, shardshuffle=shardshuffle,
            handler=wds.warn_and_continue).repeat().shuffle(shuffle)
        print(f'Loading webdataset with {len(dset.pipeline[0].urls)} shards.')

        dset = (
            dset.select(self.filter_keys).decode('pil',
                                                 handler=wds.warn_and_continue).select(self.filter_size).map_dict(
                jpg=image_transforms, handler=wds.warn_and_continue))
        for process in process_list:
            dset = dset.map(process)
        dset = (dset.batched(self.batch_size, partial=False, collation_fn=dict_collation_fn))

        loader = wds.WebLoader(dset, batch_size=None, shuffle=False, num_workers=self.num_workers)

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


if __name__ == '__main__':
    from omegaconf import OmegaConf

    config = OmegaConf.load("configs/pl_train/coadapter-v1-train.yaml")
    datamod = WebDataModuleFromConfig(**config["data"]["params"])
    dataloader = datamod.train_dataloader()

    from basicsr.utils import tensor2img
    import cv2
    save_root = 'tmp/coadapter'
    os.makedirs(save_root, exist_ok=True)

    for idx, batch in enumerate(dataloader):
        print(batch.keys())
        print(batch['jpg'].shape, torch.min(batch['jpg']), torch.max(batch['jpg']))
        img = tensor2img(batch['jpg'])
        cv2.imwrite(f'{save_root}/{idx:03d}.png', img)
        if idx > 20:
            break
