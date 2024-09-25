
import time
import torch
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tools.misc import AverageMeter
from dataset.cal_count import *
from test import obo_thres, obo_thres_list

def trainepoch(epoch, trainloader, model, optimizer, device, loss_dict, scaler, configs):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    trainLosses = []
    trainLossSSE = []
    trainOBO = []
    trainMAE = []
    traintReCo = []

    end_time = time.time()
    model.train()
    for batch_idx, batch_dict in enumerate(trainloader):

        data, target_start, target_tsm, masks= batch_dict['features'], batch_dict['target_s'], batch_dict['gt_tsm'], [batch_dict['tsm_masks'], batch_dict['masks'].squeeze(-1)]
        data_time.update(time.time() - end_time)

        with autocast():   
            optimizer.zero_grad()
            acc = 0
            data = data.type(torch.FloatTensor).to(device)
            density_start = target_start.type(torch.FloatTensor).to(device) 
            target_tsm = target_tsm.type(torch.FloatTensor).to(device)

            output_start, _, out_tsm = model(data, masks, configs['train']['sim_mode'], configs['train']['feat_norm'])


            if configs['train']['gt_mode'] == 'one_hot':
                count = torch.sum(target_start, dim=1).to(device)
                predict_count = torch.sum(output_start >= configs['threshold'], dim=1)
            elif configs['train']['gt_mode'] == 'start':
                count = count_peaks(target_start).to(device)
                predict_count = count_peaks(output_start).to(device)
            elif configs['train']['gt_mode'] == 'periodicty':
                count = torch.sum(target_start, dim=1).to(device)
                predict_count = torch.sum(output_start, dim=1).to(device)
            
            predict_density = output_start
            
            loss_sse = loss_dict['lossSSE'](predict_density, density_start)
            
            loss_mae = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                    predict_count.flatten().shape[0]  # mae
            if configs['train']['tsm_error']:
                # ignore the 0 in target tsm: out_tsm * (target_tsm == 0.0).logical_not().float()
                loss_tReCo = loss_dict['losstReCo'](out_tsm * (target_tsm == 0.0).logical_not().float(), target_tsm)  # masked TSM
            else:
                loss_tReCo = torch.tensor(np.array(0.0)).cuda()
            
            loss = loss_sse + configs['train']['lambda4tsm'] * loss_tReCo
            if configs['train']['mae_error']:
                loss += loss_mae

            # calculate MAE or OBO
            gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
            for item in gaps:
                if abs(item) <= 1:
                    acc += 1
            OBO = acc / predict_count.flatten().shape[0]
            MAE = loss_mae.item()
            
            trainOBO.append(OBO)
            trainMAE.append(MAE)
            trainLosses.append(loss.item())
            trainLossSSE.append(loss_sse.item())
            traintReCo.append(loss_tReCo.item())

            # vis
            # if batch_idx % 40 == 0:
            #     grid_gt = torchvision.utils.make_grid(batch_dict['gt_tsm'][0].cpu().unsqueeze(0), normalize=False)
            #     grid_pred = torchvision.utils.make_grid(out_tsm[0].cpu().unsqueeze(0), normalize=False)
            #     grid = torch.cat((grid_gt, grid_pred), 1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_idx % 10 == 0:
                print('iter[{}/{}], train MAE[{:.4f}], train OBO[{:.4f}], train loss[{:.4f}], train SSE[{:.4f}], train tReCo[{:.4f}], t_b[{:5.2f}/{:5.2f}], t_d[{:5.2f}]/{:5.2f}]'.format(
                  batch_idx, epoch, np.mean(trainMAE), np.mean(trainOBO), np.mean(trainLosses), np.mean(trainLossSSE), np.mean(traintReCo), batch_time.val, batch_time.avg, data_time.val, data_time.avg))
        # break

    return trainLosses, np.mean(trainMAE), np.mean(trainOBO), np.mean(traintReCo)

def validepoch(epoch, validloader, model, device, loss_dict, configs):
    validLosses = []    
    validLossSSE = []
    validOBO = []
    validMAE = []
    validtReCo = []

    with torch.no_grad():
        model.eval()
        for batch_idx, batch_dict in enumerate(validloader):
            data, target_start, target_tsm, masks = batch_dict['features'], batch_dict['target_s'], batch_dict['target_e'], batch_dict['gt_tsm'], [batch_dict['tsm_masks'], batch_dict['masks'].squeeze(-1)]

            acc = 0
            data = data.type(torch.FloatTensor).to(device)
            density_start = target_start.type(torch.FloatTensor).to(device) 
            target_tsm = target_tsm.type(torch.FloatTensor).to(device)
            
            output_start, _, out_tsm = model(data, masks, configs['train']['sim_mode'], configs['train']['feat_norm'])

            if configs['train']['gt_mode'] == 'one_hot':
                count = torch.sum(target_start, dim=1).to(device)
                predict_count = torch.sum(output_start >= configs['threshold'], dim=1)
            elif configs['train']['gt_mode'] == 'start_end':
                count = count_peaks(target_start).to(device)
                predict_count = count_peaks(output_start).to(device)
            elif configs['train']['gt_mode'] == 'periodicty':
                count = torch.sum(target_start, dim=1).to(device)
                predict_count = torch.sum(output_start, dim=1).to(device)
            
            predict_density = output_start
            

            loss_sse = loss_dict['lossSSE'](predict_density, density_start)
                
            loss_mae = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                    predict_count.flatten().shape[0]  # mae

            if configs['train']['tsm_error']:
                loss_tReCo = loss_dict['losstReCo'](out_tsm * (target_tsm == 0.0).logical_not().float(), target_tsm)  # masked TSM
            else:
                loss_tReCo = torch.tensor(np.array(0.0)).cuda()
            
            loss = loss_sse + configs['train']['lambda4tsm'] * loss_tReCo
            if configs['train']['mae_error']:
                loss += loss_mae
            
            
            gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
            for item in gaps:
                if abs(item) <= 1:
                    acc += 1
            OBO = acc / predict_count.flatten().shape[0]
        
            validOBO.append(OBO)
            validMAE.append(loss_mae.item())
            validLosses.append(loss.item())
            validLossSSE.append(loss_sse.item())
            validtReCo.append(loss_tReCo.item())
        

    return validLosses, np.mean(validMAE), np.mean(validOBO), np.mean(validtReCo)
