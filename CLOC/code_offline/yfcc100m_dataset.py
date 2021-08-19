import torch
from torchvision.datasets.vision import StandardTransform
from torch.utils.data import Dataset, IterableDataset
from PIL import Image

import os
import os.path
import csv

import sys
import math
import random
import matplotlib.pyplot as plt


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')



class YFCC_CL_Dataset_offline_val(Dataset):
    def __init__(self, args, loader = default_loader, extensions=IMG_EXTENSIONS, transform=None,
                 target_transform=None):
        
        fname = args.data_val
        root = args.root
        
        print("YFCC_CL dataset loader = {}; extensions = {}".format(loader, extensions))
        
        sys.stdout.flush()

        if isinstance(fname, torch._six.string_classes):
            fname = os.path.expanduser(fname)
        self.fname = fname

        self.transform = transform
        self.target_transform = target_transform

        self.labels, self.time_taken, self.user, self.store_loc = self._make_data(self.fname, root = root)
        if len(self.labels) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.fname)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.root = root
        self.batch_size = torch.cuda.device_count()*args.batch_size
        print("root = {}; time_taken (an example) = {}; time_taken.len = {}; batch_size = {}".format(root, self.time_taken[1000], len(self.time_taken), self.batch_size))

    def _make_data(self, fname, root):
        # read data
        fval = open(fname, 'r')
        lines_val = fval.readlines()
        labels = [None] * len(lines_val)
        time = [None] * len(lines_val)
        user = [None] * len(lines_val)
        store_loc = [None] * len(lines_val)
        
        for i in range(len(lines_val)):
            line_splitted = lines_val[i].split(",")
            labels[i] = int(line_splitted[0])
            time[i] = int(line_splitted[2])
            user[i] = line_splitted[3]
            store_loc[i] = line_splitted[-1][:-1]
        return labels, time, user, store_loc

    def __getitem__(self, index):
        if self.root is not None:
            path = self.root + self.store_loc[index]
        else:
            path = self.store_loc[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.labels[index], self.time_taken[index], index


    def __len__(self):
        return len(self.labels)



class YFCC_CL_Dataset_offline_train(Dataset):
    def __init__(self, args, loader = default_loader, extensions=IMG_EXTENSIONS, transform=None,
                 target_transform=None):
        
        fname = args.data
        root = args.root
        
        print("YFCC_CL dataset loader = {}; extensions = {}".format(loader, extensions))
        
        sys.stdout.flush()

        if isinstance(fname, torch._six.string_classes):
            fname = os.path.expanduser(fname)
        self.fname = fname

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        self._make_data()
        self.used_data_start = int(len(self.labels)*args.used_data_rate_start)
        self.used_data_end = min(len(self.labels), int(len(self.labels)*args.used_data_rate_end))
        self.data_size = self.used_data_end-self.used_data_start + 1
        self.data_size_per_epoch = int(args.num_passes*self.data_size)//int(args.epochs)

        if len(self.labels) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.fname)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.root = root
        self.batch_size = torch.cuda.device_count()*args.batch_size
        print("root = {}; time_taken (an example) = {}; time_taken.len = {}; batch_size = {}; self.used_data_start = {}; self.used_data_end = {}; self.data_size_per_epoch = {}".format(root, self.time[1000], len(self.time), self.batch_size, self.used_data_start, self.used_data_end, self.data_size_per_epoch))

    def _make_data(self):
        # only read labels and times to save storage
        self.labels = torch.load(self.fname+'train_labels.torchSave')
   #      labels = [None] * len(labels)
        self.time = torch.load(self.fname+'train_time.torchSave')
        self.user = [None] * len(self.labels)
        self.store_loc = [None] * len(self.labels)
        self.idx_data = []


    def _change_data_range(self, idx_data = None):
        if idx_data is None:
            # generate idx_data
            idx_data = self.used_data_start+torch.randperm(self.data_size)[:self.data_size_per_epoch]
        self.idx_data = idx_data
        # read user and store_locs
        self.user = [None] * len(self.labels)
        self.store_loc = [None] * len(self.labels)

        tmp_user = torch.load(self.fname+'train_user.torchSave')
        tmp_loc = torch.load(self.fname+'train_store_loc.torchSave')
        print("change data range to {}/{}".format(self.idx_data.min(), self.idx_data.max()))
        for i in range(len(self.idx_data)):
            self.idx_data[i] = min(len(self.labels)-1, self.idx_data[i])
            self.user[self.idx_data[i]] = tmp_user[self.idx_data[i]]
            self.store_loc[self.idx_data[i]] = tmp_loc[self.idx_data[i]][:-1]

    def __getitem__(self, index):
        if self.root is not None:
            path = self.root + self.store_loc[self.idx_data[index]]
        else:
            path = self.store_loc[self.idx_data[index]]
        sample = self.loader(path)

            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.labels[self.idx_data[index]], self.time[self.idx_data[index]], self.idx_data[index]


    def __len__(self):
        return len(self.idx_data)


