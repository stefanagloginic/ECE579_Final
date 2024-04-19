import os
import json
import numpy as np
import torchvision
import logging
import traceback
from torch.utils.data import Dataset


class DogPoseDataSet(Dataset):
    def __init__(self, images_dir, np_split_file, annotations_json_file, transform=None, np_skip_file=None):
        split_np = np.load(np_split_file)
        json_file = open(annotations_json_file)
        annotations = json.load(json_file)

        self.annotations = annotations
        self.transform = transform
        self.split_np = split_np
        self.root_dir = images_dir
        self.skip_np = None

        if (np_skip_file is not None):
            self.skip_np = np.load(np_skip_file)
            self.skip_np = set(self.skip_np.flatten())
            clean_split_np = []

            for index in self.split_np:
                if (index not in self.skip_np):
                    clean_split_np.append(index)

            self.split_np = clean_split_np

    def __len__(self):
        return len(self.split_np)

    def __getitem__(self, index):
        img_index = self.split_np[index]
        img_annotations = self.annotations[img_index]
        img_path = img_annotations['img_path']
        img_name = os.path.join(self.root_dir,
                                img_path)
        image = None


        try:
            image = torchvision.io.read_image(img_name, mode=torchvision.io.ImageReadMode.RGB)
        except Exception:
            logging.log(logging.WARN, "DogPoseDataSet Error Loading Image: %s", img_name)

            
        sample = {
            'image': image, 
            'img_bbox': img_annotations['img_bbox'], 
            'joints': img_annotations['joints'], 
            'img_path': img_path,
            'img_index': img_index,
            'is_multiple_dogs': img_annotations['is_multiple_dogs'],
            'seg': img_annotations['seg'],
            'img_height': img_annotations['img_height'],
            'img_width': img_annotations['img_width']
        }

        if self.transform:
            sample = self.transform(sample)

        return sample