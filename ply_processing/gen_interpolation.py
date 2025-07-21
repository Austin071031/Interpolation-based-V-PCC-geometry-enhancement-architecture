import os, glob
import numpy as np
from plyfile import *
import pandas as pd
import torch
from knn_cuda import KNN
import random

def prepare_data(save_dir, k):
    print('======!!!=====', k, '=======!!!=====')
    files = sorted(glob.glob(save_dir + k + '/*.ply'))

    groundtruth = []
    noisyinput = []
    i = 0
    for f in files:
        i += 1
        print(i, "/", len(files))

#         f2 = f.replace('noisyinput', 'ground')
        ##            temp = f2.split('rec_')
        ##            f2 = temp[0] + temp[1]

        ##                pcd = opn3d.io.read_point_cloud(f)
        ##                pcd2 = opn3d.io.read_point_cloud(f2)

        with open(f, 'rb') as f:
            plydata = PlyData.read(f)
            length = len(plydata.elements[0].data)
            data = plydata.elements[0].data
            data_pd = pd.DataFrame(data)
            coords_noisy = np.zeros((data_pd.shape[0], 3), dtype=np.float32)
            colors_noisy = np.zeros((data_pd.shape[0], 3), dtype=np.float32)
            property_names = data[0].dtype.names

        coords_noisy[:, 0] = data_pd['x']
        coords_noisy[:, 1] = data_pd['y']
        coords_noisy[:, 2] = data_pd['z']
#         colors_noisy[:, 0] = data_pd['red']
#         colors_noisy[:, 1] = data_pd['green']
#         colors_noisy[:, 2] = data_pd['blue']

#         with open(f2, 'rb') as f:
#             plydata = PlyData.read(f)
#             length = len(plydata.elements[0].data)
#             data = plydata.elements[0].data
#             data_pd = pd.DataFrame(data)
#             coords_gt = np.zeros((data_pd.shape[0], 3), dtype=np.float32)
#             colors_gt = np.zeros((data_pd.shape[0], 3), dtype=np.float32)
#             property_names = data[0].dtype.names

#         coords_gt[:, 0] = data_pd['x']
#         coords_gt[:, 1] = data_pd['y']
#         coords_gt[:, 2] = data_pd['z']
#         colors_gt[:, 0] = data_pd['red']
#         colors_gt[:, 1] = data_pd['green']
#         colors_gt[:, 2] = data_pd['blue']

        xyz = np.asarray(coords_noisy)
        rgb = np.asarray(colors_noisy)
#         xyz2 = np.asarray(coords_gt)
#         rgb2 = np.asarray(colors_gt)

        if xyz.shape[0] == 0:
            print("File NOT Copied !!")
            continue

#         ground = np.concatenate((xyz2, rgb2), axis=1)
        noisy = np.concatenate((xyz, rgb), axis=1)
#         groundtruth.append(ground)
        noisyinput.append(noisy)
        print("File Copied !!")

    return noisyinput, files

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3],如batch=8,输入点N=1024，位置信息xyz=3
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]，返回值是采样后的中心点索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    '''构建一个tensor，用来存放点的索引值（即第n个点）'''
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)#8*512
    '''构建一个距离矩阵表，用来存放点之间的最小距离值'''
    distance = torch.ones(B, N).to(device) * 1e10 #8*1024
    '''batch里每个样本随机初始化一个最远点的索引（每个batch里从1024个点中取一个）'''
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)#type为tensor(8,)
    '''构建一个索引tensor'''
    batch_indices = torch.arange(B, dtype=torch.long).to(device)#type为tensor(8,)
    for i in range(npoint):
        centroids[:, i] = farthest #第一个采样点选随机初始化的索引
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)#得到当前采样点的坐标 B*3
        dist = torch.sum((xyz - centroid) ** 2, -1)#计算当前采样点与其他点的距离，type为tensor(8,1024)
        mask = dist < distance#选择距离最近的来更新距离（更新维护这个表）
        distance[mask] = dist[mask]#将新的距离值更新到表中
        '''重新计算得到最远点索引（在更新的表中选择距离最大的那个点）'''
        farthest = torch.max(distance, -1)[1]#max函数返回值为value,index，因此取[1]值，即索引值，返回最远点索引
    return centroids

def EuclideanDist(v1, v2):
    dist = 0.0
    for i in range(len(v1)):
      dist+=abs(v1[i]-v2[i])**2

    return dist**(1/2)

def knn(point, pc, k):
    """
        for point, find the k nearest points in pc
        parameters:
            pc: np.array, [x, y, z, r, g, b]
            pc2: np.array, [[x, y, z, r, g, b]]
        return:
            returns a patch which contains the k nearest points in pc 
    """
    nearest = np.zeros((k+1, 6)) # return patch
    nearest[0] = point
    kdistance = np.zeros(k) # keep  k nearest distance
    for i in range(k):
        minDist = EuclideanDist(point[:3], pc[i][:3])
        if minDist == 0: # exclude point itself
            continue
        else:            
            nearest[i+1] = pc[i]
            kdistance[i] = minDist

    for j in range(len(pc[k:])):
        dist = EuclideanDist(point[:3], pc[j][:3])
        if dist == 0:
            continue
        else:
            if dist < np.max(kdistance): # replace the max in k-nearest
                row = np.where(kdistance==np.max(kdistance))
                kdistance[row] = dist
                nearest[row[0]+1] = pc[j]
    return nearest

def square_distance(src, dst):
 
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
 
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
 
    return dist
 
def query_ball_point(radius, nsample, xyz, new_xyz):
 
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
 
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
 
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
 
    return group_idx

if __name__ == '__main__':
    interpolation_dir_noisyinput = '/home/jupyter-austin2/HY_dataset/data_202505/r3/interpolate50/test_dataset/'
    noisy, filename = prepare_data('/home/jupyter-austin2/HY_dataset/data_202505/r3/cube70_nosmall/', 'test_dataset')

    scale_factor = 0.5
    
    for i in range(len(noisy)):
        print("="*20 + str(i+1) + ' / ' + str(len(noisy))) 
        point_cloud = noisy[i]
        point_cloud = point_cloud.reshape(1, point_cloud.shape[0], point_cloud.shape[1])
        point_cloud_tensor = torch.tensor(point_cloud).cuda()

        xyz = noisy[i][:, :3]
        xyz = xyz.reshape(1, xyz.shape[0], xyz.shape[1])
        xyz = torch.tensor(xyz).cuda()
        
        point_cloud_length = xyz.shape[1]
        NumberofInterpolate = round(scale_factor * point_cloud_length)
        
#         print(noisy[i].shape)
        print(filename[i])
        print('================ fps sampling ================')
        # fps sampling using coordinates
        centroids = farthest_point_sample(xyz, NumberofInterpolate) 
        centroids = centroids.cpu().detach()
        cent_idx = centroids[0].numpy()
        # take out the sampled points in point cloud
        querypoint = np.zeros((1, len(cent_idx), 6), dtype=np.float32)       
        for j in range(len(centroids[0])):
            querypoint[0][j] = noisy[i][cent_idx[j]]
            
#         # QPB -- use sampled point as center to group patch
#         xyz_tensor = torch.from_numpy(noisy[i]).reshape(1, noisy[i].shape[0], noisy[i].shape[1])
#         newxyz_tensor = torch.from_numpy(querypoint)
#         patch_idx = query_ball_point(100, 511, xyz_tensor, newxyz_tensor)
#         print(querypoint)
#         print(querypoint.shape)
        print('================ fine neighbors ================')
        knn=KNN(k=3,transpose_mode=True)
        querypoint_tensor = torch.tensor(querypoint).cuda()
        _, idx = knn(point_cloud_tensor, querypoint_tensor)
        
        query_with_neighbor = np.zeros((querypoint.shape[1], 3, 6), dtype=np.float32)
        
        for k in range(querypoint.shape[1]):
            query_with_neighbor[k][0] = querypoint[0][k]
            query_with_neighbor[k][1] = noisy[i][idx[0, k, 1]]
            query_with_neighbor[k][2] = noisy[i][idx[0, k, 2]]
        
        
        print('================ interpolation ================')
#         points = set()
        interpolation_xyz = np.zeros((1, query_with_neighbor.shape[0], 6), dtype=np.float32)
        for o in range(query_with_neighbor.shape[0]):
            interpolation_xyz[0][o][0] = random.randint(min(query_with_neighbor[o][0][0], query_with_neighbor[o][1][0], query_with_neighbor[o][2][0]), max(query_with_neighbor[o][0][0], query_with_neighbor[o][1][0], query_with_neighbor[o][2][0]))
            interpolation_xyz[0][o][1] = random.randint(min(query_with_neighbor[o][0][1], query_with_neighbor[o][1][1], query_with_neighbor[o][2][1]), max(query_with_neighbor[o][0][1], query_with_neighbor[o][1][1], query_with_neighbor[o][2][1]))
            interpolation_xyz[0][o][2] = random.randint(min(query_with_neighbor[o][0][2], query_with_neighbor[o][1][2], query_with_neighbor[o][2][2]), max(query_with_neighbor[o][0][2], query_with_neighbor[o][1][2], query_with_neighbor[o][2][2]))
#             interpolation_xyz[0][o][3] = random.randint(min(query_with_neighbor[o][0][3], query_with_neighbor[o][1][3], query_with_neighbor[o][2][3]), max(query_with_neighbor[o][0][3], query_with_neighbor[o][1][3], query_with_neighbor[o][2][3]))
#             interpolation_xyz[0][o][4] = random.randint(min(query_with_neighbor[o][0][4], query_with_neighbor[o][1][4], query_with_neighbor[o][2][4]), max(query_with_neighbor[o][0][4], query_with_neighbor[o][1][4], query_with_neighbor[o][2][4]))
#             interpolation_xyz[0][o][5] = random.randint(min(query_with_neighbor[o][0][5], query_with_neighbor[o][1][5], query_with_neighbor[o][2][5]), max(query_with_neighbor[o][0][5], query_with_neighbor[o][1][5], query_with_neighbor[o][2][5]))
            
#             interpolation_xyz[0][o][0] = round(random.uniform(min(query_with_neighbor[o][0][0], query_with_neighbor[o][1][0], query_with_neighbor[o][2][0]), max(query_with_neighbor[o][0][0], query_with_neighbor[o][1][0], query_with_neighbor[o][2][0])), 2)
#             interpolation_xyz[0][o][1] = round(random.uniform(min(query_with_neighbor[o][0][1], query_with_neighbor[o][1][1], query_with_neighbor[o][2][1]), max(query_with_neighbor[o][0][1], query_with_neighbor[o][1][1], query_with_neighbor[o][2][1])), 2)
#             interpolation_xyz[0][o][2] = round(random.uniform(min(query_with_neighbor[o][0][2], query_with_neighbor[o][1][2], query_with_neighbor[o][2][2]), max(query_with_neighbor[o][0][2], query_with_neighbor[o][1][2], query_with_neighbor[o][2][2])), 2)

#                 point_str = str((x, y, z))
#                 if point_str not in points:
#                     points.add(point_str)
#                     break
            

#         for index, point_str in enumerate(points):
#             interpolation_xyz[0][index] = np.array(eval(point_str))
#             print("interpolate xyz:", interpolation_xyz[0][index])
        print(interpolation_xyz.shape)
        
#         interpolation_xyz_tensor = torch.tensor(interpolation_xyz).cuda()
#         knn2 = KNN(k=1,transpose_mode=True)
#         _, idx2 = knn2(point_cloud_tensor[0, :, :3].reshape(1, point_cloud_tensor.shape[1], 3), interpolation_xyz_tensor)
#         interpolation_rgb = np.zeros((interpolation_xyz.shape[1], 3), dtype=np.float32)

#         for k in range(interpolation_xyz.shape[1]):
#             interpolation_rgb[k] = noisy[i][idx2[0, k, 0]][3:]
#         print(interpolation_rgb)
#         print(interpolation_rgb.shape)
#         # output patch as ply file
        print('================ output interpolation ================')
#         for q in range(len(patch_idx[0])):
#             print(str(q+1) + ' / ' + str(len(patch_idx[0])))
#             neighbor_x = noisy[i][patch_idx[0, q]]
#             center_x = noisy[i][cent_idx[q]].reshape((1, 6))
#             patch_x = np.concatenate((center_x,neighbor_x), axis=0)
            
#             neighbor_y = gt[i][patch_idx[0, q]]
#             center_y = gt[i][cent_idx[q]].reshape((1, 6))
#             patch_y = np.concatenate((center_y,neighbor_y), axis=0)
            
            
        patch_file = filename[i].split('/')[-1]
#             new_filename = patch_file.split('.')[0] + '_' + str(q) + '.ply'
        save_dir_noisy = interpolation_dir_noisyinput + patch_file
        if not os.path.exists(interpolation_dir_noisyinput):
            os.makedirs(interpolation_dir_noisyinput)

        with open(save_dir_noisy, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
            f.write('comment frame_to_world_scale 0.181731\n')
            f.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
            f.write('comment width 1023\n')
            f.write('element vertex ' + str(len(interpolation_xyz[0])) + '\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            for h in range(len(interpolation_xyz[0])):
                x = interpolation_xyz[0][h][0]
                y = interpolation_xyz[0][h][1]
                z = interpolation_xyz[0][h][2]
                r = int(interpolation_xyz[0][h][3])
                g = int(interpolation_xyz[0][h][4])
                b = int(interpolation_xyz[0][h][5])
                line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
                f.write(line + '\n')	