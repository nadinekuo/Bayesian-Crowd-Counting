from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np

class CrowdFish(data.Dataset):
    def __init__(self, root_path, method='train'):   # /IOCfish-Train-Val-Test/test
        self.root_path = root_path
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print(f"\nRoot path: {self.root_path}")
        print(f"\nLen of img list: {len(self.im_list)}\n\n")
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        ann_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(ann_path)
            #  for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            # return self.train_transform(img, keypoints)
            return img, keypoints
        elif self.method == 'val':
            keypoints = np.load(ann_path)
            # img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]    # XXXX
            #  for inputs, count, name in dataloader
            #   for inputs, count, name in self.dataloaders['val']:
            return img, len(keypoints), name   

# if __name__ == "__main__":
    # fish_dir = os.path.join("../IOCfish-Train-Val-Test", "test")   # /IOCfish-Train-Val-Test/test
    # fish_dataset = CrowdFish(root_path=fish_dir)
    # print(len(fish_dataset))