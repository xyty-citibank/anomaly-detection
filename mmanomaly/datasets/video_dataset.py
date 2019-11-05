
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
import numpy as np
import mmcv
import random


###
# All video index files are in one dir.
# N x C x T x H x W
class VideoDataset(Dataset):
    def __init__(self, cfg, transform=None):
        super(VideoDataset, self).__init__()
        self.frame_root = cfg.frame_root
        self.v_name_list = [name for name in os.listdir(self.frame_root) \
                              if os.path.isfile(os.path.join(self.frame_root, name))]
        self.v_name_list.sort()
        self.transform = transform
        self.time_steps = cfg.time_steps
        self.num_pred = cfg.num_pred
        self.input_size = cfg.scale_size

    def __len__(self):
        return len(self.idx_name_list)

    def __getitem__(self, item):
        """ get a video clip with stacked frames indexed by the (idx) """
        v_name = self.v_name_list[item]
        # frame index list for a video clip
        v_dir = os.path.join(self.frame_root, v_name)
        frame_list = os.listdir(v_dir)
        v_length = len(frame_list)
        clip_length = self.time_steps + self.num_pred
        start = random.randint(0, v_length - clip_length)
        frame_clip = frame_list[start, start + clip_length]
        # each sample is concatenation of the indexed frames
        c, w, h = self.input_size
        if self.transform:
            frames = torch.cat([self.transform(
                mmcv.imread(os.path.join(v_dir, i)) for i in frame_clip)], 1)
        else:
            tmp_frame_trans = transforms.ToTensor() # trans Tensor
            frames = torch.cat([tmp_frame_trans(
                mmcv.imread(os.path.join(v_dir, i)) for i \
                                in frame_clip).unsqueeze(0).resize(1, c, w, h)], 1)

        return item, frames



