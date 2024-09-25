
from dataset.cal_count import count_peaks_scaling
from dataset.RACdata import AdaptiveTData, TestData
from dataset.adaptive_loader import collateT_fn, collate_fn
from tools import set_seed
from model import RACnet

import os
import sys
import cv2
import yaml
import time
import math
import pathlib
import numpy as np
from tools import plot_tsm_heatmap, plot_action_start

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def test(config_path):

    set_seed(42)
    # setting
    configs = yaml.load(open(config_path), Loader=yaml.CLoader)

    root = os.path.join(os.path.expanduser('~'), configs['path']['root'])
    if configs['dataset'] == 'repcount':
        test_dataset = AdaptiveTData(root, configs['path']['test_video_dir'], configs['path']['test_label'])
        testloader = DataLoader(test_dataset, batch_size=configs['test']['batch_size'], pin_memory=False, shuffle=False, num_workers=configs['gpu']['num_workers'], collate_fn=collateT_fn)

        if configs['vis']:
            # TODO: only support the repcount yet
            src_path = pathlib.Path(__file__).resolve().parents[0]
            save_root = os.path.join(src_path, 'vis', configs['path']['checkpoint'].split('/')[-2])
            os.makedirs(save_root, exist_ok=True)

    else:
        test_dataset = TestData(root, configs['path']['test_video_dir'])
        testloader = DataLoader(test_dataset, batch_size=configs['test']['batch_size'], pin_memory=False, shuffle=False, num_workers=configs['gpu']['num_workers'], collate_fn=collate_fn)   
    
    
    model = RACnet(configs['model']['num_stages'], configs['model']['num_layers'], configs['model']['num_f_maps'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(model.to(device))

    if configs['path']['checkpoint']:
        print('loading checkpoint from: ', configs['path']['checkpoint'])
        checkpoint = torch.load(os.path.join(os.path.expanduser('~'), configs['path']['checkpoint']))

        model.load_state_dict(checkpoint['state_dict'], strict=True)
        del checkpoint

    testOBO = []
    testMAE = []
    predCount = []
    Count = []

    with torch.no_grad():
        start = time.time()
        for batch_idx, batch_dict in enumerate(testloader):

            if configs['dataset'] == 'repcount':
                data, target_start, target_tsm, masks, filename = batch_dict['features'], batch_dict['target_s'], batch_dict['gt_tsm'], [batch_dict['tsm_masks'], batch_dict['masks'].squeeze(-1)], batch_dict['video_name']
        
                count = count_peaks_scaling(target_start).to(device)

                if configs['vis']:
                    save_prefix = os.path.join(save_root, filename[0].split('.')[0])
                    plot_action_start(target_start.squeeze().numpy(), save_prefix + '_' + str(int(count.item())) + '_gt_actionstart.png')
                    plot_tsm_heatmap(batch_dict['gt_tsm'].squeeze().numpy(), save_prefix + '_ref_TSM.png')
            else:
                data, count, masks, filename = batch_dict['features'], batch_dict['count'], [batch_dict['tsm_masks'], batch_dict['masks'].squeeze(-1)], batch_dict['video_name']
                count = count.to(device)

            model.eval()
            acc = 0
            data = data.to(device)

            output_start, emb, sim = model(data, masks, configs['test']['sim_mode'], configs['test']['feat_norm'])

            predict_count = count_peaks_scaling(output_start, pro=configs['prominence']).to(device)

            if configs['dataset'] == 'repcount' and configs['vis']:
                plot_action_start(output_start.squeeze().detach().cpu().numpy(), save_prefix + '_' + str(int(predict_count.item())) + '_pred_densitymap.pdf')
                plot_tsm_heatmap(sim.squeeze().detach().cpu().numpy(), save_prefix + '_pred_tsm.png')
            
            mae = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                    predict_count.flatten().shape[0]  # mae
            gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
            
            for item in gaps:
                if abs(item) <= 1:
                    acc += 1
            obo = acc / predict_count.flatten().shape[0]

            testMAE.append(mae.item())
            testOBO.append(obo.item())

            predCount.append(predict_count.item())
            Count.append(count.item())
            print('file :{}, predict count :{}, groundtruth :{}'.format(filename[0], predict_count.item(), count.item()))

    print(time.time() - start)
    print("MAE:{0},OBO:{1}".format(np.mean(testMAE), np.mean(testOBO)))
    print("numbes of videos: ", len(Count))

if __name__ == '__main__':    
    CUDA_VISIBLE_DEVICES = 0
    config_path = sys.argv[1] # 'configs/test_RACnet.yaml'
    test(config_path)