import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as func

class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, out_pc, gt_pc):
        
        loss = np.zeros((out_pc.shape[0]))
        loss = torch.from_numpy(loss)
        for i in range(0, out_pc.shape[0]):
            sumloss = 0
            for j in range(gt_pc.shape[1]):
                sumloss += torch.pow((out_pc[i, 3:] - gt_pc[i, j, 3:]), 2)
            loss[i] = sumloss    
        loss = torch.mean(loss)
        return loss