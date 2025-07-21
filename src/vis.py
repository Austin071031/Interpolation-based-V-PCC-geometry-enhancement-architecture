#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 18:08:40 2018

@author: aakhtar
"""
m = 16 # 16 or 32
residual_blocks=True #True or False
block_reps = 1 #Conv block repetition factor: 1 or 2
# scale=2**5  #Voxel size = 1/scale
full_scale=2**6 #Input field size

import os
from re import M
from turtle import left
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
import open3d as opn3d
import numpy as np
import torch
from plyfile import *
import MinkowskiEngine as ME
from model.Network import MyNet
# import sparseconvnet as scn
from torch_cluster import fps
import pandas as pd
from pandas import Series, DataFrame
import time
use_cuda = torch.cuda.is_available()
checkpoint = '/home/jupyter-austin2/denoising_model/ME_soldier_30000_noisymodel_batchsize8_sampled/iter8000.pth'
prefix = 'ME_soldier_30000_noisymodel_batchsize8_sampled'
last_kernel_size = 5

#############
# VARIABLES
batch_size = 1  # I only made it working for this batch_size == 1
output_dim=4
dimension = 3
k = 10000

Data = '/home/jupyter-austin2/pc_dataset/ME_dataset/soldier/ply/rgb/val/noisyinput/'
file_names = glob.glob(Data+'*.ply')
save_dir = '/home/jupyter-austin2/Result/' + prefix + '/8000/' 

# gt= '/share12/home/baojingwei/code/mpeg-pcc-tmc2/workspace/vpcc_datasets/rgb/val/ground/'
# gt_names=glob.glob(gt+'*.ply')
# dic = {
#        "redandblack": "/share12/home/baojingwei/code/mpeg-pcc-tmc2/workspace/vpcc_datasets/rgb/val/ground",
#        "soldier": "/share12/home/baojingwei/code/mpeg-pcc-tmc2/workspace/vpcc_datasets/rgb/val/ground"
#        }

#############

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def augment(data,data2, ind):#data是xyz,data2是rgb,ind是最遠點採樣（fps）得到的index具體值，這個函數嵌套在下面的interference函數裡
    locs=[]
    feats=[]
    points=[]
    a = data.clone()
    b = data2.clone()
    offset=-a[ind]+full_scale/2
    a+=offset
    idxs=(a.min(1)[0]>=0)*(a.max(1)[0]<full_scale)
    a=a[idxs]
    b=b[idxs]
    a=a.int()
    b=b.float()
#     c = np.ones((len(b),1),dtype=np.float32)
    locs.append(torch.cat([a,torch.IntTensor(a.shape[0],1).fill_(0)],1))
    feats.append(torch.cat([b,torch.FloatTensor(b.shape[0],1).fill_(0)],1))
    points.append(torch.cat([a,torch.IntTensor(a.shape[0],1).fill_(0)],1))

    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    points=torch.cat(points,0)

    return {'x': [locs,feats], 'aug': offset, 'idxs': idxs, 'points':[points]}

def unaugment(data, offset):
    return data - offset

def inference(model, locs,y, index, device):#unet是訓練噪聲，locs是xyz，y是rgb
    with torch.no_grad():
        print('Visualization started')
#         unet.eval()
#         scn.forward_pass_multiplyAdd_count=0
#         scn.forward_pass_hidden_states=0
        point=[]
        logits = []
        indices_2 = []
        all_ind = np.asarray(list(range(locs.shape[0])))
        for i,ind in enumerate(index):
            batch = augment(locs,y,ind)
            if use_cuda:
                batch['x'][1]=batch['x'][1].cuda()
            coords = batch['x'][0][:, :3]
            feats = batch['x'][1][:, :3]
            
            feats = feats.cpu()
#             feats_1channels = np.zeros((1, 1, feats.shape[0]), dtype=np.float32)
#             feats_1channels[:, 0] = feats[:, 0]
            coords = ME.utils.batched_coordinates([coords])
            x = ME.SparseTensor(features=feats.cuda(), coordinates=coords.cuda())
#             feats_1channels = torch.from_numpy(feats_1channels)
    
            with torch.no_grad():
                out, out_cls = model(x, coords_T=feats, device=device, prune=True)
                
#             outcoords = out_cls.C[:, 1:]
#             outfeats = out_cls.F
#             outnumpy = np.concatenate((outcoords, outfeats), axis=1)

#             prediction=unet(batch['x'])
#             scalar = prediction[:,0][:,None]
#             vector = prediction[:,1:]
#             _,targets = vector.max(dim=1)
#             t = targets.unsqueeze(1)
#             onehot = torch.zeros(vector.shape).cuda()
#             onehot.scatter_(1, t, 1)
#             proj = vector*scalar#Jingwei:onhot->vector
#             x = proj + batch['x'][0][:,:3].cuda()
#             z= batch['points'][0][:,:3].cuda()

            x = out_cls.F.cuda()
#             x = x.reshape(x.shape[2], 1)
#             x = x.repeat(1, 3)
            z = out_cls.C[:, 1:].cuda()
            # Reverse the augmentation  
            z = unaugment(z, batch['aug'].cuda())#經緯：xyz--取消offset
            logits.append(x.cpu())
            point.append(z.cpu())
            indices_2.append(torch.from_numpy(all_ind[batch['idxs']]))
            if i%200 == 0:
                print('Batch_processed = %03d / %03d' % ( i, len(index)))
    return logits, indices_2, point

        
# Creating the Model and importing the information in pytorch now!!
print("=============== Creating the model and loading the checkpoint ===============")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyNet(last_kernel_size=last_kernel_size).to(device)
# model = PointNetfeat(in_dim=1, out_dim=1)
ckpt = torch.load(checkpoint)
model.load_state_dict(ckpt['model'], False)
# class Model(nn.Module):
#     def __init__(self):
#         nn.Module.__init__(self)
#         self.sparseModel = scn.Sequential().add(
#            scn.InputLayer(dimension,full_scale, mode=4)).add(
#            scn.SubmanifoldConvolution(dimension, 1, m, 3, False)).add(  # The 2nd input here is the dimension of the features.
#            scn.UNet(dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
#            scn.BatchNormReLU(m)).add(
#            scn.OutputLayer(dimension))
#         self.linear = nn.Linear(m, output_dim)
#     def forward(self,x):
#         x=self.sparseModel(x)
#         x=self.linear(x)
#         return x*full_scale/(2**3)   # +1 would translate to 1024

# unet=Model()
# if use_cuda:
#     unet=unet.cuda()
# unet.load_state_dict(torch.load(checkpoint))
# print('#classifer parameters %03d' % (sum([x.nelement() for x in unet.parameters()])))



#output_xyz = []
for i,f in enumerate(file_names):
    print("=============== File number  %i / %i  ==============="%(i+1,len(file_names)))
    
    pcd = opn3d.io.read_point_cloud(f)
    
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    
    xyz_tensor = torch.from_numpy(xyz)
    rgb_tensor = torch.from_numpy(rgb)
    
    pc = ME.SparseTensor(features=rgb_tensor.cuda(), coordinates=xyz_tensor.cuda())
#     xyz_tensor = torch.from_numpy(xyz)
#     xyz_tensor = xyz_tensor.int()
#     rgb_tensor = torch.from_numpy(rgb)
#     rgb_tensor = rgb_tensor.float()
    
    
#     temp = ME.SparseTensor(features=rgb_tensor.cuda(), coordinates=xyz_tensor.cuda())
    
#     xyz2 = temp.C.cpu().numpy()[:, :]
#     rgb2 = temp.F.cpu().detach().numpy()
#     plydata = PlyData.read(f)
#     length = len(plydata.elements[0].data)
#     data = plydata.elements[0].data
#     data_pd = pd.DataFrame(data)
#     xyz = np.zeros((data_pd.shape[0], 3), dtype=np.int32)
#     rgb = np.zeros((data_pd.shape[0], 3), dtype=np.float64)
#     property_names = data[0].dtype.names

#     for i, name in enumerate(property_names):
#         xyz[:, 0] = data_pd['x']
#         xyz[:, 1] = data_pd['y']
#         xyz[:, 2] = data_pd['z']
#         rgb[:, 0] = data_pd['red']
#         rgb[:, 1] = data_pd['green']
#         rgb[:, 2] = data_pd['blue']
    
    N = pc.F.shape[0]
    M = pc.C.shape[0]
    ratio = 10*2 / k

    print("=============== Extracting neighborhoods ===============")
    
    feats = pc.F.detach().cpu()
    locs = pc.C.detach().cpu()
    locs = locs.float()

   
    index = fps(locs, ratio=ratio, random_start=False)
    # ind_locs = locs[index]
    
    # _, indices = utility.gather_points_knn(locs, ind_locs, k)
    
    print("=============== Testing the neighborhoods ===============")
    logits,indices_2,point = inference(model,locs, feats, index, device)#
    cloud=[]
    for i in range(len(logits)):
        minicloud=torch.cat((point[i],logits[i]),1)
        cloud.append(minicloud)
    print('cloud=',cloud[0].shape)#經緯 : 確保同一patch下的xyz與rgb擁有一致的維度
    # print('rgb1=',logits[0].shape)
    # time.sleep(6)
    # print('rgb1560=',logits[1559].shape)
    # logits=torch.cat(logits,0)
    # print('logits.shape',logits.shape)
    # indices_2=torch.cat(indices_2,0)
    # print('indices_2.shape=',indices_2.shape)
    # point=torch.cat(point,0)
    # print('point.shape',point.shape)
    # time.sleep(6)
    print("============ Combining the results from each neighborhood to form final PC ============")
    #Jingwei:xyz-->rgb
    o = np.zeros((pc.F.shape[0], dimension))
    w = np.zeros((pc.F.shape[0], dimension))
#     v = np.zeros((xyz.shape[0], dimension))
#     u = np.zeros((xyz.shape[0], dimension))
#     print('o.shape=',o.shape)
#     print('cloud.shape=',np.array(logits).shape)
    for num in range(len(logits)):
        for j in range(dimension):
            o[:, j] += np.bincount(indices_2[num], logits[num][:,j], minlength=N)
            w[:, j] += np.bincount(indices_2[num], np.ones(len(indices_2[num])), minlength=N)           
#     for num in range(len(point)):
#         for j in range(dimension):        
#             v[:, j] += np.bincount(indices_2[num], point[num][:,j], minlength=M) 
#             u[:, j] += np.bincount(indices_2[num], np.ones(len(indices_2[num])), minlength=M) 
    print(len(np.where(w[:,0] == 0)[0]))
#     print(len(np.where(u[:,0] == 0)[0]))
#     v2 = np.divide(v,u)
    o2 = np.divide(o,w)
    o2 = o2[~np.isnan(o2).any(axis=1)]
#     v2 = v2[~np.isnan(v2).any(axis=1)]
    print('rgb.shape=',o2.shape)

#    output_xyz.append(o2)
    pcd2 = opn3d.geometry.PointCloud()
    pcd2.colors = opn3d.utility.Vector3dVector(o2)

    # b=f
    # b = "".join(b)
    # basename = os.path.basename(b)
    # path = dic[basename.split('_')[0]]
    # pcd3 = opn3d.io.read_point_cloud(os.path.join(path, basename))
    # pcd_tree = opn3d.geometry.KDTreeFlann(pcd3)
    # rgb = (np.asarray(pcd3.colors))*2550#Jingwei:colors->points

    # rgb2 = np.zeros((len(o2), dimension))
    # for c,n in enumerate(o2):
    #     [_, idx,_ ] = pcd_tree.search_knn_vector_3d(n,1)
    #     rgb2[c,:] = rgb[idx, :]

    pcd2.points = opn3d.utility.Vector3dVector(locs)
    
    opn3d.io.write_point_cloud(os.path.join(save_dir, os.path.basename(f)), pcd2, write_ascii=True)
    
#np.savez(os.path.join(save_dir, 'predicts'), output_xyz=output_xyz)
print("=============== Completed and results copied to destination ===============")
