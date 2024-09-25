
import os
import sys
import yaml
import time
import pathlib
import shutil
import numpy as np
from time import gmtime, strftime

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from tools import set_seed
from model import RACnet
from phases import trainepoch, validepoch
from dataset.RACdata import AdaptiveTData
from dataset.adaptive_loader import collateT_fn

def train(config_path):

    set_seed(2333)

    # setting
    configs = yaml.load(open(config_path), Loader=yaml.CLoader)

    is_debug = configs['path']['save'] == 'debug'

    # save
    root = os.path.join(os.path.expanduser('~'), configs['path']['root'])
    log_save = os.path.join('log', configs['path']['save'])
    ckpt_save = configs['path']['save']
    if not os.path.exists(os.path.join('../ckpt/{0}/'.format(ckpt_save))):
        os.makedirs(os.path.join('../ckpt/{0}/'.format(ckpt_save)))
    if not os.path.exists(log_save):
        os.makedirs(log_save)

    # dataset
    root = os.path.join(os.path.expanduser('~'), configs['path']['root'])
    train_dataset = AdaptiveTData(root, configs['path']['train_video_dir'], configs['path']['train_label'],gt_mode=configs['train']['gt_mode'])
    trainloader = DataLoader(train_dataset, batch_size=configs['train']['batch_size'], pin_memory=False, shuffle=True, num_workers=configs['gpu']['num_workers'], collate_fn=collateT_fn)
    valid_dataset = AdaptiveTData(root, configs['path']['valid_video_dir'], configs['path']['valid_label'], gt_mode=configs['train']['gt_mode'])
    validloader = DataLoader(valid_dataset, batch_size=configs['train']['batch_size'], pin_memory=False, shuffle=False, num_workers=configs['gpu']['num_workers'], collate_fn=collateT_fn)
    
    # model
    model = RACnet(configs['model']['num_stages'], configs['model']['num_layers'], configs['model']['num_f_maps'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(model.to(device)) 

    # optimizer
    currEpoch = 0
    if configs['train']['optim'] == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=configs['train']['lr'], weight_decay=configs['train']['weight_decay'])
    elif configs['train']['optim'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=configs['train']['lr'], weight_decay=configs['train']['weight_decay'], momentum=0.9)
    
    # interruption
    if configs['path']['checkpoint']:
        print('loading checkpoint from: ', configs['path']['checkpoint'])
        checkpoint = torch.load(configs['path']['checkpoint'])
        currEpoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'], strict=False)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        del checkpoint
    
    # pretrain
    if configs['path']['pretrain']:
        print('loading pretrain from: ', configs['path']['pretrain'])
        checkpoint = torch.load(os.path.join(os.path.expanduser('~'), configs['path']['pretrain']))
        model.load_state_dict(checkpoint['state_dict'], strict=True)

        del checkpoint
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    scaler = GradScaler()

    # lr decay
    if configs['train']['lr_decay'] == 'multisteplr':
        milestones = [i for i in range(0, configs['train']['epochs'], 40)][1:]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.9)  # three step decay 0.95
    elif configs['train']['lr_decay'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    elif configs['train']['lr_decay'] == 'cosine_warmup':
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=180 - configs['train']['warmup'], eta_min=0.00005)
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=configs['train']['warmup']
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[configs['train']['warmup']])
    elif configs['train']['lr_decay'] == 'reducelr':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.95, threshold_mode='abs')
    
    gt_mode = configs['train']['gt_mode']

    # loss
    loss_dict = {'lossSSE': nn.MSELoss(reduction='sum'),
            'losstReCo': nn.MSELoss(reduction='sum'),
            'lossL1': nn.SmoothL1Loss()}

    print('training start: ')
    print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

    best_mae = 1
    best_obo = 0
    for epoch in range(currEpoch, configs['train']['epochs']):
        # train
        epoch_start = time.time()
        trainLosses, epoch_trainMAE, epoch_trainOBO, epoch_traintReCo = trainepoch(epoch, trainloader, model, optimizer, device, loss_dict, scaler, configs)  
        print('train epoch time: {:.2f}min'.format((time.time() - epoch_start)/60.))

        if is_debug:
            epoch=51

        if configs['path']['pretrain'] or epoch > 10:
            # valid
            epoch_start = time.time()
            validLosses, epoch_validMAE, epoch_validOBO, epoch_validtReCo = validepoch(epoch, validloader, model, device, loss_dict, configs)
            print('val epoch time: {:.2f}min'.format((time.time() - epoch_start)/60.))
        else:
            validLosses = []
            epoch_validMAE = 0

        if configs['train']['lr_decay'] == 'reducelr':
            scheduler.step(epoch_validMAE)
        else:
            scheduler.step()

        
        if configs['path']['pretrain'] or epoch > 10:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainLosses': trainLosses,
                'valLosses': validLosses
            }

            # best valid results
            if epoch_validMAE < best_mae:
                best_mae = epoch_validMAE
                save_name = os.path.join( '../ckpt/{0}/best'.format(ckpt_save) + '_' + str(epoch) + '_mae_' + str(round(epoch_validMAE, 2)) + '_' + str(round(epoch_validOBO, 2)) + '.pt')
                torch.save(checkpoint, save_name)
            if epoch_validOBO > best_obo:
                best_obo = epoch_validOBO
                save_name = os.path.join( '../ckpt/{0}/best'.format(ckpt_save) + '_' + str(epoch) + '_' + str(round(epoch_validMAE, 2)) + '_obo_' + str(round(epoch_validOBO, 2)) + '.pt')
                torch.save(checkpoint, save_name)

        if configs['path']['pretrain'] or epoch > 10:
            print('epochs[{}/{}], train MAE[{:.4f}], train OBO[{:.4f}], train loss[{:.4f}], train tReCo[{:.4f}], valid MAE[{:.4f}], valid OBO[{:.4f}], valid loss[{:.4f}], valid tReCo[{:.4f}]'.format(
            epoch, configs['train']['epochs'], epoch_trainMAE, epoch_trainOBO, np.mean(trainLosses), epoch_traintReCo, epoch_validMAE, epoch_validOBO, np.mean(validLosses), epoch_validtReCo
            ))
        else:
            print('epochs[{}/{}], train MAE[{:.4f}], train OBO[{:.4f}], train loss[{:.4f}], train tReCo[{:.4f}]'.format(
            epoch, configs['train']['epochs'], epoch_trainMAE, epoch_trainOBO, np.mean(trainLosses), epoch_traintReCo
            ))


    print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

if __name__ == '__main__':

    config_path = sys.argv[1] # configs/train_RACnet.yaml
    train(config_path)