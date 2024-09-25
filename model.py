import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        for s in self.stages:
            out = s(torch.sigmoid(out) * torch.unsqueeze(mask, 1), mask) # softmax -> sigmoid
            # [num_stages, batch_size, class, seq_len]
        return torch.sigmoid(out).squeeze(-2)

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * torch.unsqueeze(mask, 1) 
        return out
    
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * torch.unsqueeze(mask, 1)

def similarity_matrix(emb, mode, tsm_mask):
    """
    x: b, f, 512
    E: Euclidean distance
    C: Correlation similarity
    O: self-attention similarity
    H: haiming similarity
    COS: COSINESIMILARITY
    """
    if 'E' in mode:
        #  --- nagetive euclidien distance ---
        temperature = 1 # 13.5
        tsm = torch.cdist(emb, emb, p=2)**2
        tsm = -(tsm * tsm_mask) # RepNet: temperature = 13.5
        if mode == 'E-softmax':
            tsm /= temperature
            tsm = tsm.softmax(dim=-1)
        elif mode == 'E-sigmoid':
            tsm = tsm.sigmoid()
        elif mode == 'E-norm':
            tsm = (tsm - tsm.min(dim=2)[0].unsqueeze(-1)) / (tsm.max(dim=2)[0].unsqueeze(-1) - tsm.min(dim=2)[0].unsqueeze(-1)).add_(1.0e-8)
            
    elif mode == 'C':
        # --- correlation distance ---
        tsm = torch.matmul(emb, emb.transpose(-2, -1)) 
        scale = tsm_mask.sum(2)[:,0].unsqueeze(1).unsqueeze(1)
        tsm = tsm / torch.sqrt(scale)
        tsm = tsm.softmax(dim=-1)

    elif 'H' in mode:
        #  --- nagetive hamming distance ---
        temperature = 1
        if mode == 'H-selfnorm':
            b, f, d = emb.shape
            epsilon = 1.0e-8
            diff = torch.abs(emb.unsqueeze(2) - emb.unsqueeze(1))
            tsm = (diff > epsilon).sum(dim=-1)
            tsm = (tsm - tsm.min(dim=2)[0].unsqueeze(-1)) / (tsm.max(dim=2)[0].unsqueeze(-1) - tsm.min(dim=2)[0].unsqueeze(-1)).add_(1.0e-8)

        if mode == 'H-norm':
            tsm = torch.cdist(emb, emb, p=0)**2
            tsm = -((tsm * tsm_mask) / temperature)
            tsm = (tsm - tsm.min(dim=2)[0].unsqueeze(-1)) / (tsm.max(dim=2)[0].unsqueeze(-1) - tsm.min(dim=2)[0].unsqueeze(-1)).add_(1.0e-8)

    elif 'COS' in mode:
        # ----- cosine similarity --------
        square = torch.sum(emb * emb, dim=-1) # b,f
        tsm = torch.matmul(emb, emb.transpose(-2, -1)) / ((square ** 0.5).unsqueeze(-1) * (square.unsqueeze(1) ** 0.5)).add_(1.0e-8)
        tsm = tsm * tsm_mask
        if mode == 'COS-norm':
            tsm = (tsm - tsm.min(dim=2)[0].unsqueeze(-1)) / (tsm.max(dim=2)[0].unsqueeze(-1) - tsm.min(dim=2)[0].unsqueeze(-1)).add_(1.0e-8)
        if mode == 'COS-sig':
            tsm = tsm.sigmoid()
    else:
        print("No mode of calculate TSM: {}".format(mode))
    
    return tsm * tsm_mask

def feature_norm(emb, mask, mode='min'):
    # x: b, 512, f
    # feature norm
    # 1. x-min/max-min
    if mode == 'min-max':
        emb = (emb - emb.min(dim=-2, keepdim=True)[0]) / (emb.max(dim=-2, keepdim=True)[0] - emb.min(dim=-2, keepdim=True)[0])
    # 2. -mean / std Standardization
    elif mode == 'std':
        emb = (emb - emb.mean(dim=-2, keepdim=True)) / emb.std(dim=-2, keepdim=True)
    # 3. feature L2 norm
    elif mode == 'l2':
        emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    else:
        return emb 
    return emb * mask

class FinetuneF(nn.Module):
    def __init__(self):
        super(FinetuneF, self).__init__()
        self.conv3D = nn.Conv3d(in_channels=768,
                                out_channels=512,
                                kernel_size=3,
                                padding=(3, 1, 1),
                                dilation=(3, 1, 1))
        self.bn1 = nn.BatchNorm3d(512)
        self.SpatialPooling = nn.MaxPool3d(kernel_size=(1, 7, 7))
        self.drop = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
    
    def forward(self, x, mask):
        # feature -> [b,f,768*7*7] offline feature
        num_frames = x.shape[-2]
        input_size = x.shape
        features = self.drop(x.view(-1, num_frames, 768, 7, 7))

        x = features.transpose(1, 2)
        x = F.relu(self.bn1(self.conv3D(x)))  # ->[b,512,f,7,7]
        x = self.SpatialPooling(self.drop2(x))  # ->[b,512,f,1,1]
        x = x.squeeze(3).squeeze(3)  # -> [b,512,f]
        assert x.shape[2] == mask.shape[1], 'the size should match, but now size input:{}, size output:{}, size mask{}'.format(input_size, x.shape, mask.shape)
        return x * torch.unsqueeze(mask, 1)

class RACnet(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, temperature=1):
        super(RACnet, self).__init__()
        self.finetune_feature = FinetuneF()
        self.multistage = MultiStageModel(num_stages, num_layers, num_f_maps, 512, 1)
        self.temperature = temperature
        self.tsm = similarity_matrix
        self.feature_normalization = feature_norm

    def forward(self, x, masks, sim_mode, feat_norm_mode):
        tsm_mask, mask = masks
        x = self.finetune_feature(x, mask) # -> b, 512, f

        emb = self.feature_normalization(x, mask.unsqueeze(1), feat_norm_mode)

        tsm = self.tsm(emb.transpose(-2, -1), sim_mode, tsm_mask)

        out = self.multistage(x, mask) # x: [B, 512, F] mask: [B, F] # B, F

        return out, emb, tsm
