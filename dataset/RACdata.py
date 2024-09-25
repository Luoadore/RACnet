"""
AdaptiveTData: RepCount data loader from different frames file. 
TestData: UCFRep and Countix data loader.
Adaptive fashion.
"""

import os
import os.path as osp
import numpy as np
import math
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from .gen_gt import get_labels_dict, GenerateGT


class TestData(Dataset):

    """mainly for testing, these data don't have fine grained annotations, just counts"""

    def __init__(self, root_path, video_path):
        """
        :param root_path: root path
        :param video_path: video child path (folder)
        :param label_path: label child path(.csv)
        """
        self.root_path = root_path
        self.video_path = os.path.join(self.root_path, video_path)  # train or valid
        self.video_dir = os.listdir(self.video_path)
        
    def __getitem__(self, inx):
        """
        get data item
        :param video_tensor, label
        """
        video_file_name = self.video_dir[inx]
        file_path = os.path.join(self.video_path, video_file_name)
        video_tensor, video_frame_length, label = self.get_frames(file_path)
        return [video_tensor, label, video_file_name, video_frame_length]
    
    def __len__(self):
        """:return the number of video """
        return len(self.video_dir)

    def get_frames(self, npz_path):
        # get frames from .npz files
        with np.load(npz_path, allow_pickle=True) as data:
            frames = data['img_feature']  # numpy.narray [frames_length, feat_dim]
            frames_length = len(frames)
            count = data['count']  # numpy.narray [frames_length]

            frames_length = frames.shape[0]  
            frames = torch.FloatTensor(frames)
            label = torch.FloatTensor(np.array(float(count)))
            print(label, type(label))
        return frames, frames_length, label


class AdaptiveTData(Dataset):

    def __init__(self, root_path, video_path, label_path, gt_mode='start'):
        """
        :param root_path: root path
        :param video_path: video child path (folder)
        :param label_path: label child path(.csv)
        """
        self.root_path = root_path
        self.video_path = os.path.join(self.root_path, video_path)  # train or valid
        self.label_path = os.path.join(self.root_path, label_path)
        self.video_dir = os.listdir(self.video_path)
        self.label_dict = get_labels_dict(self.label_path)  # get all original labels
        self.gt_mode = gt_mode
        
    def __getitem__(self, inx):
        """
        get data item
        :param video_tensor, label
        """
        video_file_name = self.video_dir[inx]
        file_path = os.path.join(self.video_path, video_file_name)
        video_tensor, video_frame_length, video_tsm_tensor = self.get_frames(file_path)
        if video_file_name.replace('_feature.npz', '.npz') in self.label_dict.keys():
            labels = self.label_dict[video_file_name.replace('_feature.npz', '.npz')]
            generator = GenerateGT() 
            prob_label_start, prob_label_end = generator.gen_gt(labels, video_frame_length, self.gt_mode)
            prob_label_start = torch.FloatTensor(prob_label_start)
            prob_label_end = torch.FloatTensor(prob_label_end)
            prob_label = (prob_label_start, prob_label_end)
            
            return [video_tensor, video_tsm_tensor, prob_label, video_file_name] # [frame_length, feat_dim], [frame_length, frame_length], ([frame_length], [frame_length])
        else:
            print(video_file_name, 'the label of this video does not exist')
            return
    
    def __len__(self):
        """:return the number of video """
        return len(self.video_dir)

    def get_frames(self, npz_path):
        # get frames from .npz files
        with np.load(npz_path, allow_pickle=True) as data:
            frames = data['frames']  # numpy.narray [frames_length, feat_dim]
            frames_length = data['length']
            gt_tsm = data['gt_tsm'] # numpy.narray [frames_length, frames_length]
              
            frames = torch.FloatTensor(frames)
            gt_tsm = torch.FloatTensor(gt_tsm)
            
        return frames, frames_length, gt_tsm