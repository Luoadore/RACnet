"""Adaptive for dataloader when it comes to train"""

import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import ZeroPad2d

def collate_fn(data, pad_ignore_idx=-100):
    """
    collate function for test data
    """
    video_tensor, label_tensor, video_filename_tensor, video_frame_length = zip(*data)

    batch_video_tensor = pad_sequence(video_tensor, batch_first=True) # [B, F, D]
    
    masks = torch.ones(batch_video_tensor.shape[:2]) # [B, F]
    masks = masks[:, :, None].bool() # [B, F, 1]
    tsm_masks = torch.ones(masks.shape[0], masks.shape[1], masks.shape[1]).bool() # [B, F, F]

    video_filename_tensor = np.array(video_filename_tensor)

    video_tensor = torch.stack(video_tensor)
    label_tensor = torch.stack(label_tensor)

    batch_dict = {"features": batch_video_tensor,
                  "count": label_tensor,
                  "video_name": video_filename_tensor,
                  "video_frame_lenth": video_frame_length,
                  "masks": masks,
                  "tsm_masks": tsm_masks
                  }

    return batch_dict

def collateT_fn(data, pad_ignore_idx=-100):
    """
    adapt the length to get the batch data using padding, padding both ahead and after
    including the TSM gt
    """
    video_tensor, video_tsm_tensor, label_tensor, video_filename_tensor = zip(*data) # [frame_length, feat_dim], [frame_length, frame_length], ([frame_length], [frame_length])

    batch_video_tensor = pad_sequence(video_tensor, batch_first=True) # B*[Fi, D] ->[B, F, D]
    
    batch_target_start_tensor = pad_sequence([label_tensor[i][0] for i in range(len(label_tensor))], batch_first=True, padding_value=pad_ignore_idx) # [B, F]
    batch_target_end_tensor = pad_sequence([label_tensor[i][1] for i in range(len(label_tensor))], batch_first=True, padding_value=pad_ignore_idx) # [B, F]

    masks = torch.where(batch_target_start_tensor == pad_ignore_idx, 0, 1) # [B, F]
    masks = masks[:, :, None].bool() # [B, F, 1]

    batch_tsm_masks_tensor = []
    pad_length = batch_target_end_tensor.shape[1]
    batch_video_tsm_tensor = []
    for each in video_tsm_tensor:
        each_len = each.shape[1]
        padding = ZeroPad2d((0, pad_length - each_len, 0, pad_length - each_len))
        batch_video_tsm_tensor.append(padding(each))
        batch_tsm_masks_tensor.append(padding(torch.ones((each_len, each_len))).bool())

    batch_video_tsm_tensor = torch.stack(batch_video_tsm_tensor, dim=0) #[B, F, F]
    batch_tsm_masks_tensor = torch.stack(batch_tsm_masks_tensor, dim=0) #[B, F, F]
    

    video_filename_tensor = np.array(video_filename_tensor)

    batch_dict = {"features": batch_video_tensor,
                  "target_s": batch_target_start_tensor * masks.squeeze(-1),
                  "target_e": batch_target_end_tensor * masks.squeeze(-1),
                  "gt_tsm": batch_video_tsm_tensor,
                  "video_name": video_filename_tensor,
                  "masks": masks,
                  "tsm_masks": batch_tsm_masks_tensor}

    return batch_dict