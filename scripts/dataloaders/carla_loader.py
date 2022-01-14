import os
import os.path
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
from dataloaders import transforms
from dataloaders import CoordConv


def load_calib():
    calib = open("dataloaders/carla_camera_K.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[0]
    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    K = np.reshape(np.array([float(p) for p in Proj_str]),
                   (3, 3)).astype(np.float32)
    # Resize data into 224*224
    K[0][0] = K[0][0] * (224 / 1024)
    K[0][2] = 224 / 2
    K[1][1] = K[1][1] * (224 / 576)
    K[1][2] = 224 / 2
    return K


def get_paths_and_transform(split, args):
    if split == "train":
        transform = train_transform
        glob_d = os.path.join(args.data_folder, 'train/sparsedepmap/*.png')
        glob_gt = os.path.join(args.data_folder, 'train/depth/*.png')
        glob_rgb = os.path.join(args.data_folder, 'train/rgb/*.jpg')

    elif split == "val":
        transform = no_transform
        glob_d = os.path.join(args.data_folder, 'val/sparsedepmap/*.png')
        glob_gt = os.path.join(args.data_folder, 'val/depth/*.png')
        glob_rgb = os.path.join(args.data_folder, 'val/rgb/*.jpg')

    elif split == "test":
        if args.test == "easy":
            transform = no_transform
            glob_d = os.path.join(args.data_folder, 'test_easy/sparsedepmap/*.png')
            glob_gt = os.path.join(args.data_folder, 'test_easy/depth/*.png')
            glob_rgb = os.path.join(args.data_folder, 'test_easy/rgb/*.jpg')
        elif args.test == "middle":
            transform = no_transform
            glob_d = os.path.join(args.data_folder, 'test_middle/sparsedepmap/*.png')
            glob_gt = os.path.join(args.data_folder, 'test_middle/depth/*.png')
            glob_rgb = os.path.join(args.data_folder, 'test_middle/rgb/*.jpg')
        elif args.test == "hard":
            transform = no_transform
            glob_d = os.path.join(args.data_folder, 'test_hard/sparsedepmap/*.png')
            glob_gt = os.path.join(args.data_folder, 'test_hard/depth/*.png')
            glob_rgb = os.path.join(args.data_folder, 'test_hard/rgb/*.jpg')
        elif args.test == "hardest":
            transform = no_transform
            glob_d = os.path.join(args.data_folder, 'test_hardest/sparsedepmap/*.png')
            glob_gt = os.path.join(args.data_folder, 'test_hardest/depth/*.png')
            glob_rgb = os.path.join(args.data_folder, 'test_hardest/rgb/*.jpg')
    else:
        raise ValueError("Unrecognized split " + str(split))

    paths_d = sorted(glob.glob(glob_d))
    paths_gt = sorted(glob.glob(glob_gt))
    paths_rgb = sorted(glob.glob(glob_rgb))

    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths, transform


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    depth = depth_png.astype(np.float) / 256.
    depth = np.expand_dims(depth, -1)
    return depth


def train_transform(rgb, sparse, target, position, args):
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
    transforms_list = [transforms.HorizontalFlip(do_flip),
                       transforms.Resize((224, 224))]
    transform_geometric = transforms.Compose(transforms_list)
    sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    brightness = np.random.uniform(max(0, 1 - args.jitter),
                                   1 + args.jitter)
    contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
    saturation = np.random.uniform(max(0, 1 - args.jitter),
                                   1 + args.jitter)
    transform_rgb = transforms.Compose([
        transforms.ColorJitter(brightness, contrast, saturation, 0),
        transform_geometric
    ])
    rgb = transform_rgb(rgb)
    return rgb, sparse, target, position


def no_transform(rgb, sparse, target, position, args):
    transforms_list = [transforms.Resize((224, 224))]
    transform_geometric = transforms.Compose(transforms_list)

    sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    rgb = transform_geometric(rgb)
    return rgb, sparse, target, position


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


class CarlaDepth(data.Dataset):
    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
        self.K = load_calib()
        self.threshold_translation = 0.1

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index])
        sparse = depth_read(self.paths['d'][index])
        target = depth_read(self.paths['gt'][index])
        return rgb, sparse, target

    def __getitem__(self, index):
        rgb, sparse, target = self.__getraw__(index)
        # position = CoordConv.AddCoordsNp(self.args.val_h, self.args.val_w)
        position = CoordConv.AddCoordsNp(rgb.shape[0], rgb.shape[1])
        position = position.call()
        rgb, sparse, target, position = self.transform(rgb, sparse, target, position, self.args)

        candidates = {"rgb": rgb, "d": sparse, "gt": target, 'position': position, 'K': self.K}

        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths['gt'])
