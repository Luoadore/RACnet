import math
import numpy as np
from scipy.signal import find_peaks

import torch

# find_peaks(x, prominence) -> peaks, {‘prominences’, ‘right_bases’, ‘left_bases’}
def count_peaks(sequence, pro=0.3):
    """
    calculate the peaks of counting
    sequence: numpy [batch, frame_length, 1]
    """
    sequence = sequence.cpu().detach().numpy()
    prominence=pro
    counts = [len(find_peaks(each, prominence=prominence)[0]) for each in sequence]
    # border:
    for i, each in enumerate(sequence):
        if each[0] > prominence and each[0] - each[1] > 0:
            counts[i] += 1
        if each[-1] > prominence and each[-1] - each[-2] > 0: 
            counts[i] += 1
    return torch.Tensor(counts)

def border(seq, count, thres):
    count_border = 0
    if seq[0] > thres and seq[0] - seq[1] > 0:
        count_border += 1
    if seq[-1] > thres and seq[-1] - seq[-2] > 0: 
        count_border += 1
    return count + count_border

def count_peaks_scaling(sequence, pro=0.2):
    # scaling all the sequence before counting
    # min max scaling
    sequence = sequence.cpu().detach()
    sequence = (sequence - sequence.min(dim=1)[0].unsqueeze(-1)) / (sequence.max(dim=1)[0].unsqueeze(-1) - sequence.min(dim=1)[0].unsqueeze(-1))
    sequence = sequence.numpy()

    counts = []
    counts = [border(each, len(find_peaks(each, prominence=pro)[0]), pro) for each in sequence]
    return torch.Tensor(counts)

