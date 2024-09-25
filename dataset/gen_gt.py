import cv2
import csv
import math
import numpy as np
import os.path as osp
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.fft import fft
import matplotlib.pyplot as plt
# from .label_norm import normalize_label

def pdf(x, mu, sig):
    # f(x)
    # Gaussian
    # without normalization
    return np.exp(-(x - mu) ** 2 / (2 * sig ** 2)) # / (math.sqrt(2 * math.pi) * sig)

def get_integrate(x_1, x_2, mu, sig):
    # area
    y, err = integrate.quad(pdf, x_1, x_2, args=(mu, sig))
    return y

class GenerateGT:
    """gen_label2prob_perboundary"""
    def __init__(self, mean=0, sigma=1, mode='start'):
        self.mean = mean
        self.sigma = sigma
        self.mode = mode # mode = ['start', 'periodicity']
        self.local_pdf_half, self.half_len = self.get_local_pdf(self.mean, self.sigma)
    
    def get_local_pdf(self, mean, sigma, left=-2, right=3):
        
        x = np.arange(left, right)
        local_gaussian_value = [pdf(x[i], mean, sigma) for i in range(len(x))]
        local_gaussian_value_half = local_gaussian_value[:len(x) // 2 + 1]
        gaus_half_len = len(local_gaussian_value_half)

        return local_gaussian_value_half, gaus_half_len


    def local2globel(self, local, local_len, labels, frame_len):

        label = [0 for _ in range(frame_len)]

        for i in labels:

            if i == 0:
                for j in range(local_len):
                    if label[i + j] != 0:
                        print('the label already generate: index[{}], privious:[{}], new:[{}]'.format(i + j, label[i + j], local[local_len - j - 1]))
                    else:
                        label[i + j] = local[local_len - j - 1]

            elif i == len(label) - 1:
                for j in range(local_len):
                    if label[i - j] != 0:
                        print('the label already generate: index[{}], privious:[{}], new:[{}]'.format(i - j, label[i - j], local[local_len - j - 1]))
                    else:
                        label[i - j] = local[local_len - j - 1]

            else:
                for j in range(local_len):
                    if i + j >= len(label):
                        break
                    if label[i + j] != 0:
                        print('the label already generate: index[{}], privious:[{}], new:[{}]'.format(i + j, label[i + j], local[local_len - j - 1]))
                    else:
                        label[i + j] = local[local_len - j - 1]
                    if label[i - j] != 0 or i + j == i - j:
                        if i + j != i - j:
                            print('the label already generate: index[{}], privious:[{}], new:[{}]'.format(i - j, label[i - j], local[local_len - j - 1]))
                    else:
                        label[i - j] = local[local_len - j - 1]
        
        return label

    def one_hot(self, labels, frame_len):
        label = [1 if i in set(labels) else 0 for i in range(frame_len)]
        return label

    ### comparsion with Transrac, the normalize_label func can be found in the official repository of TransRac
    # def periodicty_label(self, labels, frame_len):
    #     label = normalize_label(labels, frame_len)
    #     return label

    def gen_gt(self, labels, frame_len, mode='start'):
        
        if mode == 'start':
            start_label = self.local2globel(self.local_pdf_half, self.half_len, [labels[i] for i in range(0, len(labels), 2)], frame_len)
            end_label = self.local2globel(self.local_pdf_half, self.half_len, [labels[i] for i in range(1, len(labels), 2)], frame_len)
            # print(start_label, end_label)

        elif mode == 'boundary_action':
            pass

        # full resolution periodicty
        # elif mode == 'periodicty':
        #     start_label = self.periodicty_label(labels, frame_len)
        #     end_label = self.periodicty_label(labels, frame_len)
        #     return (start_label, end_label)
        
        # periodicty density map totally the same as transrac
        # elif mode == 'transrac':
        #     sample_frames = 64
        #     new_crop = []
        #     for i in range(len(labels)):  # frame_length -> 64
        #         item = min(math.ceil((float((labels[i])) / float(frame_len)) * sample_frames), sample_frames - 1)
        #         new_crop.append(item)
        #     new_crop = np.sort(new_crop)
        #     label = normalize_label(new_crop, sample_frames)
        #     return (label + [0] * (frame_len - sample_frames), label + [0] * (frame_len - sample_frames))

        elif mode == 'one_hot':
            start_label = self.one_hot([labels[i] for i in range(0, len(labels), 2)], frame_len)
            end_label = self.one_hot([labels[i] for i in range(1, len(labels), 2)], frame_len)
        
        return (start_label, end_label)


def get_labels_dict(path):
    # read label.csv to RAM
    # 'video.npz': [start, end, start, end...]
    labels_dict = {}
    check_file_exist(path)
    with open(path, encoding='utf-8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            cycle = [int(float(row[key])) for key in row.keys() if 'L' in key and row[key] != '']
            if not row['count']:
                print(row['name'] + ' does not have count from the annotation.')
            else:
                labels_dict[row['name'].split('.')[0] + str('.npz')] = np.array(cycle)

    return labels_dict


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


if __name__ == '__main__':

    root_path = r''
    test_label = [88,130,130,	166,	166,	201,	201,	236,	236,	272,	272,	306,	306,	344,	344,	380,	380,	415,	415,	452,	452,	486,	486,	521,	521,	559,	559,	594,	594,	630,	631,	666,	666,	701,	701,	738,	738,	778]

    print(test_label)
    frame_length = 780

    