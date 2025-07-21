import torch
import torch.nn as nn
import numpy as np
import os,sys

sys.path.append('../')

from auction_match import auction_match
import pointnet2.pointnet2_utils as pn2_utils
import math
from knn_cuda import KNN
# from emd_cal.emd_module import emdModule


class Loss(nn.Module):
    def __init__(self,radius=1.0):
        super(Loss,self).__init__()
        self.radius=radius
        self.knn_repulsion=KNN(k=1,transpose_mode=True)
#     def get_emd_loss(self,pred,gt,radius=1.0):
#         '''
#         pred and gt is B N 3
#         '''
#         idx, _ = auction_match(pred.contiguous(), gt.contiguous())
#         #gather operation has to be B 3 N
#         #print(gt.transpose(1,2).shape)
#         matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
# #         print(matched_out.device())
#         matched_out = matched_out.transpose(1, 2).contiguous()
# #         print(matched_out.device())
#         dist2 = (pred - matched_out) ** 2
#         dist2 = dist2.view(dist2.shape[0], -1)  # <-- ???
#         dist2 = torch.mean(dist2, dim=1, keepdims=True)  # B,
# #         print(dist2.device())
#         dist2 /= radius
#         return torch.mean(dist2)
    
    def get_repulsion_loss(self,pcd,h=0.0005):
        dist,idx=self.knn_repulsion(pcd,pcd)#B N k

        dist=dist[:,:,1:5]**2 #top 4 cloest neighbors

        loss=torch.clamp(-dist+h,min=0)
        loss=torch.mean(loss)
        #print(loss)
        return loss
    
    def get_mse_loss(self,pred, gt):
        dist,idx=self.knn_repulsion(gt,pred)#B N k
        mse_loss = torch.nn.MSELoss()
        loss = 0
        
        align_gt = torch.zeros(pred.shape)

        for i in range(pred.shape[1]):
            align_gt[0][i] = gt[0][idx[0][i][0]]

        
        B2A = mse_loss(pred, align_gt.cuda())
        
        loss += B2A
        
        return loss

    def get_mse_loss2(self,pred, gt, weight):
        dist,idx=self.knn_repulsion(gt,pred)#B N k
        mse_loss = torch.nn.MSELoss()
        loss = 0
        
        align_gt = torch.zeros(pred.shape)

        for i in range(pred.shape[1]):
            align_gt[0][i] = gt[0][idx[0][i][0]]

        
        B2A = mse_loss(pred, align_gt.cuda())
        
        loss += B2A * weight
        
        _, indx2 = self.knn_repulsion(pred, gt)
    
        align_output = torch.zeros(gt.shape)
        for j in range(gt.shape[1]):
            align_output[0][j] = pred[0][indx2[0][j][0]]

        A2B = mse_loss(align_output.cuda(), gt)
        loss += A2B * (1 - weight)
#         #print(loss)
        return loss

#     def get_emd_loss2(self, pred, gt, device):
#         emd_loss = emdModule()
#         if pred.shape[1] % 1024 != 0:
#             pad_num = pred.shape[1] // 1024 + 1
#             pad_pred = torch.nn.functional.pad(pred, [0, 0, 0, pad_num*1024 - pred.shape[1]])
#         else:
#             pad_pred = pred
#         allign_KNN = KNN(k=1,transpose_mode=True)
#         dist,idx=allign_KNN(gt,pad_pred)#B N k

#         align_gt = torch.zeros(pad_pred.shape)

#         for i in range(pred.shape[1]):
#             align_gt[0][i] = gt[0][idx[0][i][0]]

#         dis, assignment = emd_loss(pad_pred, align_gt, 0.005, 50)
#         loss = torch.tensor(np.sqrt(dis.detach().cpu().numpy()).mean()).to(device)
        
#         return loss
    
    def get_color_mse_loss(self, pred_color, noisy_xyz, gt_color, gt_xyz):
        noisy_xyz = noisy_xyz.unsqueeze(0)
        gt_xyz = gt_xyz.unsqueeze(0)
        pred_color = pred_color.unsqueeze(0)
        gt_color = gt_color.unsqueeze(0)
        
        
        
        dist,idx=self.knn_repulsion(gt_xyz,noisy_xyz)#B N k
        
        align_gt_color = torch.zeros(pred_color.shape)

        for i in range(pred_color.shape[1]):
            align_gt_color[0][i] = gt_color[0][idx[0][i][0]]

        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(pred_color, align_gt_color.cuda())
#         #print(loss)
        return loss
    
if __name__=="__main__":
    loss=Loss().cuda()
    point_cloud=torch.rand(1,2048,3).cuda()
    gt=torch.rand(1,200000,3).cuda()
    emd_loss=loss.get_emd_loss(point_cloud, gt)
    repulsion_loss=loss.get_repulsion_loss(point_cloud)
    mse_loss = loss.get_mse_loss(point_cloud, gt)
    
    print("EMD_Loss: ", emd_loss)
    print("Repulsion_Loss: ", repulsion_loss)
    print("MSE_Loss: ", mse_loss)