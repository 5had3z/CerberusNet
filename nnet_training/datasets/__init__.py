import os
import platform
import multiprocessing
from typing import Dict

import torch
from torch.utils.data import RandomSampler

from .cityscapes_dataset import CityScapesDataset
from .kitti_dataset import Kitti2015Dataset
from .custom_batch_sampler import BatchSamplerRandScale
from .custom_batch_sampler import collate_w_bboxes

__all__ = ['get_dataset']

def id_vec_generator(train_ratio, directory):
    """
    Generates the training and validation split of a monlitic dataset.\n
    Ensures that you are always using the same testing and training data for a given set and ratio.
    """
    num_images = 0
    for file in os.listdir(directory):
        num_images += file.endswith('.png')

    print(f"Number of Images:\t{num_images}")
    n_train = int(num_images * train_ratio)

    train_ids = list(range(n_train))
    val_ids = list(range(n_train, num_images))

    return train_ids, val_ids

def get_dataset(dataset_config) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Input configuration json for dataset
    Output dataloaders for training and validation
    """
    if platform.system() == 'Windows':
        n_workers = 0
    else:
        n_workers = min(multiprocessing.cpu_count()-2, dataset_config.batch_size)

    aux_aug = {}
    if 'img_normalize' in dataset_config.augmentations:
        aux_aug['img_normalize'] = dataset_config.augmentations.img_normalize
    if 'disparity_out' in dataset_config.augmentations:
        aux_aug['disparity_out'] = dataset_config.augmentations.disparity_out
    if 'box_type' in dataset_config.augmentations:
        aux_aug['box_type'] = dataset_config.augmentations.box_type

    if dataset_config.type == "Kitti":
        # Using the segmentaiton gt dir to count the number of images
        seg_dir = os.path.join(dataset_config.rootdir, "semantic")
        train_ids, val_ids = id_vec_generator(dataset_config.train_ratio, seg_dir)

        datasets = {
            'Training'   : Kitti2015Dataset(
                dataset_config.rootdir, dataset_config.objectives,
                **dataset_config.augmentations, id_vector=train_ids),
            'Validation' : Kitti2015Dataset(
                dataset_config.rootdir, dataset_config.objectives,
                output_size=dataset_config.augmentations.output_size,
                id_vector=val_ids, **aux_aug)
        }
    elif dataset_config.type == "Cityscapes":
        datasets = {
            'Training'   : CityScapesDataset(
                dataset_config.rootdir, dataset_config.subsets, 'train',
                **dataset_config.augmentations),
            'Validation' : CityScapesDataset(
                dataset_config.rootdir, dataset_config.subsets, 'val',
                output_size=dataset_config.augmentations.output_size, **aux_aug)
        }
    else:
        raise NotImplementedError(f"Dataset not implemented: {dataset_config.type}")

    dataloaders = {
        'Validation' : torch.utils.data.DataLoader(
            datasets["Validation"],
            batch_size=dataset_config.batch_size,
            shuffle=dataset_config.shuffle,
            num_workers=n_workers,
            drop_last=dataset_config.drop_last,
            pin_memory=True,
            collate_fn=collate_w_bboxes
        )
    }

    if hasattr(dataset_config.augmentations, 'rand_scale'):
        dataloaders['Training'] = torch.utils.data.DataLoader(
            datasets["Training"], num_workers=n_workers, pin_memory=True,
            batch_sampler=BatchSamplerRandScale(
                sampler=RandomSampler(datasets["Training"]),
                batch_size=dataset_config.batch_size,
                drop_last=dataset_config.drop_last,
                scale_range=dataset_config.augmentations.rand_scale),
                collate_fn=collate_w_bboxes
        )
    else:
        torch.backends.cudnn.benchmark = True

        dataloaders['Training'] = torch.utils.data.DataLoader(
            datasets["Training"],
            batch_size=dataset_config.batch_size,
            shuffle=dataset_config.shuffle,
            num_workers=n_workers,
            drop_last=dataset_config.drop_last,
            pin_memory=True,
            collate_fn=collate_w_bboxes
        )

    return dataloaders
