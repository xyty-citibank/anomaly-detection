
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
import numpy as np
import mmcv
import random
from mmanomaly.datasets.transform import GroupImageTransform
from mmanomaly.datasets.utils import *

###
# All video index files are in one dir.
# N x C x T x H x W
class VideoDataset(Dataset):
    def __init__(self, cfg):
        super(VideoDataset, self).__init__()
        self.frame_root = cfg.data_root
        self.v_name_list = [name for name in os.listdir(self.frame_root)]
        self.v_name_list.sort()

        self.time_steps = cfg.data.train.time_steps
        self.num_pred = cfg.data.train.num_pred
        self.input_size = cfg.data.train.scale_size
        self.img_norm_cfg = cfg.img_norm_cfg
        self.transform = GroupImageTransform()

    def __len__(self):
        return len(self.v_name_list)

    def __getitem__(self, item):
        """ get a video clip with stacked frames indexed by the (idx) """
        v_name = self.v_name_list[item]
        # frame index list for a video clip
        v_dir = os.path.join(self.frame_root, v_name)
        frame_list = os.listdir(v_dir)
        v_length = len(frame_list)
        clip_length = self.time_steps + self.num_pred
        start = random.randint(0, v_length - clip_length)
        frame_clip = frame_list[start:start + clip_length]
        # each sample is concatenation of the indexed frames
        c, w, h = self.input_size
        if self.transform:
            frame_arr = []
            for i in frame_clip:
                frame_arr.append(mmcv.imread(os.path.join(v_dir, i)))

            frames, img_shape, pad_shape, scale_factor, crop_quadruple = self.transform(frame_arr, (w, h), keep_ratio=False, div_255=True)
            frames = to_tensor(frames)
            size = len(frame_clip)
            frames = frames.reshape(c * size, w, h)
        else:
            tmp_frame_trans = transforms.ToTensor() # trans Tensor

            frames = torch.cat([tmp_frame_trans(
                mmcv.imresize(mmcv.imread(os.path.join(v_dir, i)), self.scale_size)) for i in frame_clip], 0)

        return frames



