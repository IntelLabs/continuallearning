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
    """A data loader for YFCC100M dataset
    """
    def __init__(self, args, loader = default_loader, extensions=IMG_EXTENSIONS, transform=None,
                 target_transform=None):
        
        fname = args.data_val
        root = args.root
        
        print("YFCC_CL dataset loader = {}; extensions = {}".format(loader, extensions))
        
        sys.stdout.flush()

        if isinstance(fname, torch._six.string_classes):
            fname = os.path.expanduser(fname)
        self.fname = fname

        # for backwards-compatibility
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
            
        # for replay buffer
        self.batch_size = torch.cuda.device_count()*args.batch_size
        self.is_forward = True
        self.offset = 0
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

    def set_transfer_time_point(self, args, val_set, time_last, is_forward = True):
        # find idx that is larger and closest to time_last
        self.is_forward = is_forward
        for i in range(len(self.time_taken)):
            if self.time_taken[i] >= time_last:
                print("[set_transfer_time_point]: time_last = {}; time[{}] = {}".format(time_last, i, self.time_taken[i]))
                self.offset = i
                return
        self.offset = len(self.time_taken) - 1

    def __getitem__(self, index):
        if self.is_forward:
            index = min(len(self.labels), index + self.offset)
        else:
            index = max(0, self.offset - index)

        if self.root is not None:
            path = self.root + self.store_loc[index]
        else:
            path = self.store_loc[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.labels[index], self.time_taken[index], index


    def __len__(self):
        if self.is_forward:
            return len(self.labels) - self.offset
        else:
            return self.offset + 1


class YFCC_CL_Dataset_online(Dataset):

    def __init__(self, args, loader = default_loader, extensions=IMG_EXTENSIONS, transform=None, transform_RepBuf = None,
                 target_transform=None, target_transform_RepBuf = None, trans_test = None):
        
        fname = args.data
        root = args.root 
        size_buf = args.size_replay_buffer        

        print("YFCC_CL dataset loader = {}; extensions = {}".format(loader, extensions))
        
        sys.stdout.flush()

        if isinstance(fname, torch._six.string_classes):
            fname = os.path.expanduser(fname)
        self.fname = fname

        # for backwards-compatibility
        self.transform = transform
        self.transform_test = trans_test
        self.target_transform = target_transform
        self.transform_RepBuf = transform_RepBuf
        self.target_transform_RepBuf = target_transform_RepBuf

        # valid initial and final index
        self.used_data_start = args.used_data_start
        self.used_data_end = args.used_data_end

        self._make_data()
        self.data_size = len(self.labels)
        self.data_size_per_epoch = math.ceil(self.data_size/args.epochs)

        self.batch_size = torch.cuda.device_count()*args.batch_size

        # make batch size always 256
        if self.data_size_per_epoch % self.batch_size != 0:
            self.data_size_per_epoch = self.data_size_per_epoch - self.data_size_per_epoch % self.batch_size + self.batch_size    

    
        self.loader = loader
        self.extensions = extensions
        self.root = root
        self.size_buf = size_buf

        self.repBuf_sample_rate = 0.5 # use this later, test 0.5 case first
        self.sampling_strategy = args.sampling_strategy
            
        self.NOSubBatch = args.NOSubBatch
        self.SubBatch_index_offset = int(self.batch_size/self.NOSubBatch)

        self.gradient_steps_per_batch = int(args.gradient_steps_per_batch) # repBatch_rate is multiplied by this
        self.repBatch_rate = math.ceil(self.repBuf_sample_rate/(1-self.repBuf_sample_rate))

        self.repType = args.ReplayType

        print("[initData]: root = {}; time_taken (an example) = {}; time_taken.len = {}; batch_size = {}; size_buf = {}; repBuf_sample_rate = {}".format(root, self.time_taken[1000], len(self.time_taken), self.batch_size, self.size_buf, self.repBuf_sample_rate))
        print("[initData]: repBatch_rate = {}; NOSubBatch = {}; SubBatch_index_offset = {}".format(self.repBatch_rate, self.NOSubBatch, self.SubBatch_index_offset))
        print("[initData]: transform = {}; transform_RepBuf = {}".format(self.transform, self.transform_RepBuf))

    def _make_data(self):
       # only read labels and times to save storage
        self.labels = torch.load(self.fname+'train_labels.torchSave')
        self.time_taken = torch.load(self.fname+'train_time.torchSave')
        self.user = torch.load(self.fname+'train_userID.torchSave')
        self.store_loc = [None] * len(self.labels)
        self.idx_data = []

    def _change_data_range_FIFO(self):
        print("reading store location from {}".format(self.fname+'train_store_loc.torchSave'))
        tmp_loc = torch.load(self.fname+'train_store_loc.torchSave')
        print("tmp_loc.size = {}".format(len(tmp_loc)))
        for i in range(self.used_data_start, self.used_data_end):
            self.store_loc[i] = tmp_loc[i][:-1]
            if i % 1e5 == 0:
                print("store_loc[{}] = {}".format(i, self.store_loc[i]))
            sys.stdout.flush()        


    def _set_data_idx(self, epoch):
        # setup training data idx
        self.offset = epoch*self.data_size_per_epoch
        size_curr_epoch = min(self.data_size - self.offset, self.data_size_per_epoch) 
        size_curr_epoch = size_curr_epoch - (size_curr_epoch % self.batch_size)

        batchSize = int(self.batch_size/self.NOSubBatch)
        self.data_idx = [None]* (size_curr_epoch *self.gradient_steps_per_batch)
        iter_total = size_curr_epoch//batchSize
        bsReplicated = batchSize*self.gradient_steps_per_batch
        for i in range(0, int(iter_total)):
            self.data_idx[i*bsReplicated:(i+1)*bsReplicated] = list(range(self.offset+i*batchSize, self.offset+(i+1)*batchSize))*self.gradient_steps_per_batch
            

    def _change_data_range(self, epoch = 0):
        self._set_data_idx(epoch)

        self.used_data_start = max(0, epoch*self.data_size_per_epoch-self.size_buf-self.batch_size)
        self.used_data_end = min(len(self.labels), (epoch+1)*self.data_size_per_epoch+self.batch_size)
        print("change valid data range to: [{},{}]".format(self.used_data_start, self.used_data_end))
        
        self.store_loc = [None] * len(self.labels)
        self._change_data_range_FIFO()


    def _sample_FIFO(self, index):
        if index < self.batch_size:
            return 0
        else:
            repBuf_idx = random.randint(max(0, index-self.size_buf-self.batch_size), index-self.batch_size)
            return repBuf_idx


    def _sample(self, index):
        return self._sample_FIFO(index)        

    def __getitem__(self, index):
        index = self.data_idx[index] 
        index_pop = index

        if index_pop < 0:
            index_pop = 0
            is_valid = torch.tensor(0)
        else:
            is_valid = torch.tensor(1)

        num_batches = 1
    
        target_pop = self.labels[index_pop]
        path_pop = self.root + self.store_loc[index_pop]
        sample_pop = self.loader(path_pop)
        if self.transform is not None:
            sample_pop = self.transform(sample_pop)
        if self.target_transform is not None:
            target_pop = self.target_transform(target_pop)


        sample_test = torch.zeros(self.NOSubBatch, sample_pop.size()[0], sample_pop.size()[1], sample_pop.size()[2])
        target_test = torch.zeros(self.NOSubBatch).long()
        test_idx = torch.zeros(self.NOSubBatch).long()      
        time_taken_test = torch.zeros(self.NOSubBatch).long()
        user_test = torch.zeros(self.NOSubBatch).long()

        # get the test data of full batch size
        for i in range(0, self.NOSubBatch):
            test_idx[i] = index+i*self.SubBatch_index_offset
            if test_idx[i] >= len(self.time_taken):
                test_idx[i] = len(self.time_taken) - 1

            time_taken_test[i] = self.time_taken[test_idx[i]]
            target_test[i] = self.labels[test_idx[i]]
            user_test[i] = self.user[test_idx[i]]
            
            path = self.root + self.store_loc[test_idx[i]]
            sample = self.loader(path)
            if self.transform is not None:
                sample_test[i] = self.transform_test(sample)
            if self.target_transform is not None:
                target_test[i] = self.target_transform(target_test[i])

        # randomly sample from repBuf
        sample_RepBuf = torch.zeros(self.repBatch_rate* num_batches, sample_pop.size()[0], sample_pop.size()[1], sample_pop.size()[2])
        target_RepBuf = torch.zeros(self.repBatch_rate* num_batches).long()
        repBuf_idx = torch.zeros(self.repBatch_rate * num_batches).long()        

        for i in range(0, int(self.repBatch_rate * num_batches)):
            repBuf_idx[i] = self._sample(index)
            path_RepBuf = self.root + self.store_loc[repBuf_idx[i]]
            target_RepBuf[i] = self.labels[repBuf_idx[i]]
           
            sample_RepBuf_tmp = self.loader(path_RepBuf)
            if self.transform_RepBuf is not None:
                # print("performing to sample_RepBuf: {}".format(self.transform_RepBuf))
                sample_RepBuf[i] = self.transform_RepBuf(sample_RepBuf_tmp)
            if self.target_transform_RepBuf is not None:
                target_RepBuf[i] = self.target_transform_RepBuf(target_RepBuf[i])

        return sample_test, target_test, user_test, time_taken_test, test_idx, sample_pop, target_pop, index_pop, sample_RepBuf, target_RepBuf, repBuf_idx

    def __len__(self):
        return len(self.data_idx)
