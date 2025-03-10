"""Run the RacNet model on a given video."""

import time
import os
import csv
import cv2
import torch
import torch.nn as nn
import numpy as np

from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint

from model.RACnet import RACnet
from dataset.cal_count import count_peaks_scaling


def get_dict(root, video_path, anno_path):
    # dict filename: label
    data_dict = {}
    labels_dict = {}
    video_lists = os.listdir(os.path.join(root, video_path))
    if 'RepCountA' in video_path:
        with open(os.path.join(root, anno_path), encoding='utf-8') as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                if not row['count']:
                    print(row['name'] + ' does not have count from the annotation.')
                else:
                    labels_dict[row['name']] = int(row['count'])
    for each in video_lists:
        data_dict[each] = labels_dict[each]
    return data_dict

def load_model(config, checkpoint):
        # # # load  pretrained model of video swin transformer using mmaction and mmcv API
        cfg = Config.fromfile(config)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

        # # # load hyperparameters by mmcv api
        load_checkpoint(model, checkpoint, map_location='cpu')
        backbone = model.backbone

        print('--------- backbone loaded ------------')

        return backbone

def get_item(video_path):
    # video
    cap = cv2.VideoCapture(video_path)
    frames = []
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if success is False:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (224, 224))
            frames.append(frame_rgb)
    cap.release()
    frames = np.asarray(frames)  # [f,w,h,c]
    if (frames.size != 0):
        frames = frames.transpose(0, 3, 2, 1)  # [f,c,h,w]
    else:
        print(video_path, ' is wrong video. size = 0')
        return -1

    frames = torch.FloatTensor(frames)
    frames -= 127.5
    frames /= 127.5
    frames_length = len(frames)

    # masks
    mask = torch.ones(1, frames_length).bool() # [1, F]
    tsm_mask = torch.ones(1, mask.shape[1], mask.shape[1]).bool() # [1, F, F]
    masks = [tsm_mask, mask]

    return frames.transpose(0, 1).unsqueeze(0), masks


if __name__ == '__main__':
    ### ----- configs -------
    # data
    root = os.path.expanduser('~') 
    video_path = 'path/to/your/data/'
    anno_path = 'path/to/test.csv'
    
    # model
    num_stages = 4
    num_layers = 10
    num_f_maps = 64
    pretrain_model = os.path.join(root, 'path/to/pretrain.pt')
    extractor_config = os.path.join(root, 'configs/swin_tiny_patch244_window877_kinetics400_1k.py')
    checkpoint = os.path.join(root, 'path/to/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth')

    ### -------- test ------------
    # Dataset
    data_dict = get_dict(root, video_path, anno_path)

    # Load model
    device = torch.device("cuda") 

    feature_extractor = load_model(extractor_config, checkpoint)
    feature_extractor.eval()
    feature_extractor.to(device)

    model = RACnet(num_stages, num_layers, num_f_maps)
    model = nn.DataParallel(model.to(device))
    state_dict = torch.load(pretrain_model)
    model.load_state_dict(state_dict['state_dict'], strict=True)
    model.eval()
    model.to(device)
    del state_dict

    # test
    inference_time = []
    testOBO = []
    testMAE = []
    predCount = []
    Count = []
    testOBRatio_e1 = [] 
    with torch.no_grad():
        for each in data_dict.keys():
            
            v_path, count = each, np.array(data_dict[each])
            video_name = video_path.split('/')[-1].split('.')[0]

            count = torch.FloatTensor(np.array([count]))

            # download the video sample
            frames, masks = get_item(os.path.join(root, video_path, v_path))
            frames = frames.to(device)

            # run model
            start = time.time()

            frames = [frames[:, :, i:i + 1, :, :] for i in range(0, frames.shape[2])]
            slice = []
            for frame in frames:
                frame = feature_extractor(frame)
                slice.append(frame)
            feature = torch.cat(slice, dim=2).transpose(1, 2)
            out, emb, tsm = model(feature, masks, 'E-norm', 'None')

            out = (out - out.min(dim=1)[0].unsqueeze(-1)) / (out.max(dim=1)[0].unsqueeze(-1) - out.min(dim=1)[0].unsqueeze(-1))

            predict_count = count_peaks_scaling(out, pro=0.2).to(device)
            infer = time.time() - start

            count = count.to(device)

            mae = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                        predict_count.flatten().shape[0]  # mae
            gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
            acc = 0
            for item in gaps:
                if abs(item) <= 1:
                    acc += 1
            obo = acc / predict_count.flatten().shape[0]
            testOBO.append(obo)

            MAE = mae.item()
            testMAE.append(MAE)
            predCount.append(predict_count.item())
            Count.append(count.item())
            inference_time.append(infer)
            print('file :{}, predict count :{}, groundtruth :{}, time :{}'.format(video_name, predict_count.item(), count.item(), infer))

    
    print("MAE:{0},OBO:{1}".format(np.mean(testMAE), np.mean(testOBO)))
    print("numbes of videos: ", len(Count))
    print("avg inference time: {} / {} = {}".format(np.sum(inference_time), len(Count), np.sum(inference_time) / len(Count)))