import cv2
import csv
import numpy as np
from scipy.interpolate import interp1d

class GenerateTSM:
    def __init__(self):
        """
        Generate the reference TSM from the original annotations
        """
        pass

    def line2grid(self, xinterp, yinterp, eachx, eachy):
        # line to small mesh grid
        small_grid = np.zeros((eachy[0] - eachx[0], eachy[1] - eachx[1]))
        for sx, sy in zip(xinterp, yinterp):
            # sy should be ceil and floor
            sy = [np.ceil(sy) - eachx[1], np.floor(sy) - eachx[1]]
            for each_sy in sy:
                if each_sy >= small_grid.shape[1]:
                    each_sy = small_grid.shape[1] - 1
                small_grid[sx][int(each_sy)] = 1.0
        return small_grid

    def smooth(self, tsm, filter, stride, mode='8neighbors'):

        h, w = tsm.shape
        filter_h, filter_w = filter.shape
        stride_h, stride_w = stride

        padding_h = (filter_h - 1) // 2
        padding_w = (filter_w - 1) // 2
        padding_data = np.zeros((h + padding_h * 2, w + padding_w * 2))
        padding_data[padding_h:-padding_h, padding_w:-padding_w] = tsm

        smooth_tsm = np.zeros((h // stride_h, w // stride_w))
        
        for idx_h, i in enumerate(range(0, h, stride_h)):
            for idx_w, j in enumerate(range(0, w, stride_w)):
                window = padding_data[i: i + filter_h, j: j + filter_w]
                if mode == 'conv':
                    smooth_tsm[idx_h, idx_w] = np.sum(window * filter)
                if mode == '8neighbors':
                    smooth_tsm[idx_h, idx_w] = max(window[filter_h // 2, filter_w // 2], 0.8 * np.max(window))
                if mode == '8neighbors_expo':
                    max_r, max_c = np.where(window == window.max())
                    distance = np.array([np.abs(x - filter_h // 2) + np.abs(y - filter_w // 2) for x, y in zip(max_r, max_c)])
                    smooth_tsm[idx_h, idx_w] = max(window[filter_h // 2, filter_w // 2], math.pow(0.8, np.min(distance)) * np.max(window))

        return smooth_tsm

    def norm(self, tsm, mode='scale'):
        zmin, zmax = np.nanmin(tsm), np.nanmax(tsm)
        if mode == 'normalize':
            norm_tsm = (tsm - zmin) / (zmax - zmin)
        if mode == 'scale':
            norm_tsm = tsm / zmax
        return norm_tsm

    def gen_tsm(self, labels, frame_len, filter_size=9, smooth_mode='8neighbors_expo'):
        """
        gt: [(start, end)]
        frame_len: length of video raw frames        
        """
        tsm = np.eye(frame_len)
        start_gt = [labels[i] for i in range(0, len(labels), 2)]
        end_gt = [labels[i] for i in range(1, len(labels), 2)]
        points = [(start, end) for (start, end) in zip(start_gt, end_gt)]

        # --- endpoint ---
        # each start should totally similar: 1
        for each_start in start_gt:
            for each in start_gt:
                tsm[each_start][each] = 1.0
        # same as end
        for each_end in end_gt:
            for each in end_gt:
                tsm[each_end][each] = 1.0 

        # --- two endpoints of the line and fullfill the tsm ---
        for i in range(len(points) - 1):
            # find endpoints-pair
            mainp = points[i]
            restp = points[i+1:]

            endpointx = [(mainp[0], each[0]) for each in restp]
            endpointy = [(mainp[1], each[1]) for each in restp]

            # draw line
            # interpolate
            for eachx, eachy in zip(endpointx, endpointy):
                # print(eachx, eachy)
                x = [eachx[0], eachy[0]]
                y = [eachx[1], eachy[1]]
                fx = interp1d(x, y, kind='linear')

                # line to grid
                xinterp = np.arange(eachx[0], eachy[0])
                yinterp = fx(xinterp)
                small_grid = self.line2grid(xinterp - eachx[0], yinterp, eachx, eachy)

                # put small grid back to tsm
                tsm[eachx[0]:eachy[0], eachx[1]:eachy[1]] = small_grid
                small_grid_flip = np.rot90(np.flip(small_grid, axis=0), 1)
                tsm[eachx[1]:eachy[1], eachx[0]:eachy[0]] = small_grid_flip

        # --- smooth tsm --- 
        smooth_tsm = self.smooth(tsm, np.ones((filter_size, filter_size)), np.array([1, 1]), mode=smooth_mode)
        norm_tsm = smooth_tsm
        
        return norm_tsm

    def vis_tsm(self, save_path, tsm):
        tsm = np.clip(tsm * 255, 0, 255).astype(np.uint8)
        tsm = cv2.applyColorMap(tsm, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(save_path, tsm)

def get_frames_length(path):
    # get the frames length of the raw video
    frames_dict = {}
    with open(path, encoding='utf-8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            frames_dict[row['filename']] = int(row['length'])
    return frames_dict

if __name__ == '__main__':
    
    # an example of how to generate reference TSM
    from gen_gt import get_labels_dict

    label_path = 'path_to_annotation'
    frame_length_path = '../metadata/RepCountA_frame_length.csv'
    
    refTSM = GenerateTSM()
    labels_dict = get_labels_dict(label_path)
    frame_length_dict = get_frames_length(frame_length_path) 
    for each in labels_dict.keys():
        labels = labels_dict[each]
        frame_length = frame_length_dict[each]
        reftsm = refTSM.gen_tsm(labels, frame_length)

