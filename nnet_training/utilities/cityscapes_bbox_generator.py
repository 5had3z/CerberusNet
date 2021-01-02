#!/usr/bin/env python3

"""
Generates bboxes for cityscapes dataset using instance and semantic annotations from cityscapes \
dataset.
"""

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

import os
import json
import multiprocessing
from typing import List, Dict, Union

import cv2
import numpy as np
from PIL import Image

from nnet_training.utilities.visualisation import CITYSPALLETTE
from nnet_training.utilities.cityscapes_labels import label2trainid, trainId2name

def cs_seg_gen(path_: str) -> str:
    """
    Generator that yields all relevant image paths in the dataset
    """
    for dirpath, _, filenames in os.walk(path_):
        for filename in filenames:
            if filename.endswith('gtFine_instanceIds.png'):
                yield os.path.join(dirpath, filename)

def visualise_output(base_img: np.ndarray,
    bbox_list: List[Dict[str, Union[int, np.ndarray]]]) -> None:
    """
    Shows overlay of bboxes generated and semantic segmentation gt.
    """
    overlay_img = np.zeros_like(base_img)

    for bbox_info in bbox_list:
        bbox = bbox_info['bbox']
        overlay_img = cv2.rectangle(
            overlay_img, tuple(bbox[:2]), tuple(bbox[2:]), (255, 0, 0), thickness=2)
        cv2.putText(
            overlay_img, trainId2name[bbox_info['train_id']], tuple(bbox[:2]),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.add(overlay_img, base_img, dst=overlay_img)

    cv2.imshow("bbox test", overlay_img)
    cv2.waitKey(0)

def seg_cv2_bgr(path_: str) -> np.ndarray:
    """
    Returns RGB image for gt labeled image segmentation
    """
    seg_img = np.array(Image.open(path_), dtype=np.int16)

    keys = np.array([255, 255, 255, 255, 255, 255,
                     255, 255, 0, 1, 255, 255,
                     2, 3, 4, 255, 255, 255,
                     5, 255, 6, 7, 8, 9,
                     10, 11, 12, 13, 14, 15,
                     255, 255, 16, 17, 18])
    mapping = np.array(range(-1, len(keys) - 1)).astype('int32')

    values = np.unique(seg_img)
    for value in values:
        assert value in mapping
    index = np.digitize(seg_img.ravel(), mapping, right=True)
    seg_img = keys[index].reshape(seg_img.shape)

    color_lut = np.zeros((256, 3), dtype=np.uint8)
    color_lut[:len(CITYSPALLETTE)//3, :] = np.asarray(
        [CITYSPALLETTE[i:i + 3] for i in range(0, len(CITYSPALLETTE), 3)],
        dtype=np.uint8)

    seg_frame = np.empty(shape=(seg_img.shape[0], seg_img.shape[1], 3), dtype=np.uint8)
    for j in range(3):
        seg_frame[..., j] = cv2.LUT(seg_img.astype(np.uint8), color_lut[:, j])
    cv2.cvtColor(seg_frame, cv2.COLOR_RGB2BGR, dst=seg_frame)

    return seg_frame

def bbox_info_from_instance(inst_img: np.ndarray) -> List[Dict[str, Union[int, np.ndarray]]]:
    """
    Given a pixel-wise instance label of image, generate a list of bboxes.\n
    Non instance-wise labels are id'd as 0-23. Instanced labels are labeled as xxyyy, \
    where xx is the class id and yyy is the unique number for that instance.
    """
    bbox_dict = {}
    for x_idx in range(0, inst_img.shape[1]):
        for y_idx in range(0, inst_img.shape[0]):
            inst_id = int(inst_img[y_idx, x_idx])
            if inst_id in bbox_dict.keys():
                bbox_dict[inst_id][0] = min(bbox_dict[inst_id][0], x_idx)
                bbox_dict[inst_id][1] = min(bbox_dict[inst_id][1], y_idx)
                bbox_dict[inst_id][2] = max(bbox_dict[inst_id][2], x_idx)
                bbox_dict[inst_id][3] = max(bbox_dict[inst_id][3], y_idx)
            elif inst_id > 23:
                bbox_dict[inst_id] = [x_idx, y_idx, x_idx, y_idx]

    return [{'id': inst_id // 1000,
            'train_id' : label2trainid[inst_id // 1000],
            'bbox' : bbox}
            for inst_id, bbox in bbox_dict.items()]

def write_info_json(inst_path: str, bbox_info: List[Dict[str, Union[int, np.ndarray]]]) -> None:
    """
    Writes bbox info to json file
    """
    json_path = inst_path.replace('instanceIds.png','bbox.json')
    with open(json_path, 'w') as bbox_file:
        json.dump(bbox_info, bbox_file)

def test_image():
    """
    Tests algo on single image and shows output.
    """
    src_path = '/media/bryce/1TB Samsung/ml_datasets/cityscapes_data/gtFine/train/aachen'
    inst_path = os.path.join(src_path, 'aachen_000000_000019_gtFine_instanceIds.png')
    seg_path = os.path.join(src_path, 'aachen_000000_000019_gtFine_labelIds.png')

    inst_img = np.array(Image.open(inst_path), dtype=np.int16)

    base_img = seg_cv2_bgr(seg_path)

    bboxes = bbox_info_from_instance(inst_img)

    write_info_json(inst_path, bboxes)

    visualise_output(base_img, bboxes)

def generate_bbox_json(path_: str) -> None:
    """
    Given a path to the instance segmentation, creates and writes the bbox info json.
    """
    inst_img = np.array(Image.open(path_), dtype=np.int16)
    bboxes = bbox_info_from_instance(inst_img)
    write_info_json(path_, bboxes)

def main(path_: str) -> None:
    """
    Traverses segmentation ground truth folder and generates the bbox info json file for each.
    """
    print(f"Parsing {path_}")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(generate_bbox_json, cs_seg_gen(path_))
    print("Done!")

if __name__ == "__main__":
    main('/media/bryce/1TB Samsung/ml_datasets/cityscapes_data/gtFine')
