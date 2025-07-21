import torch
import numpy as np
from sklearn.neighbors import KDTree
from plyfile import *
import numpy as np
import pandas as pd
import os

def k_nearest_neighbors(A, B, K):
    # convert numpy arrays to PyTorch tensors
#     A = torch.from_numpy(A).float()
#     B = torch.from_numpy(B).float()

    # split the position and color information
    A_pos, A_col = A[:, :3], A[:, 3:]
    B_pos, B_col = B[:, :3], B[:, 3:]
    

    # build a KD-Tree from the positions of point cloud B
    tree = KDTree(B_pos)

    # find the indices of the K nearest neighbors for each point in A
    distances, indices = tree.query(A_pos, k=K)

    # create an array to store the results
    result = np.zeros((A.shape[0], K, 6))

    # loop over each point in A
    for i in range(A.shape[0]):
        # get the indices of the K nearest neighbors
        nn_indices = indices[i]

        # gather the positions and colors of the K nearest neighbors from B
        nn_pos = B_pos[nn_indices]
        nn_col = B_col[nn_indices]
        

        # concatenate the positions and colors into a single tensor
        nn = np.concatenate((nn_pos, nn_col), axis=-1)

        # store the result for this point
        result[i] = nn

    return result

noisy_block_file_path = '/home/jupyter-austin2/pc_dataset/ME_dataset/soldier/ply_block/val/noisyinput/'
gt_file_path = '/home/jupyter-austin2/pc_dataset/ME_dataset/soldier/ply/rgb/val/ground/'
gt_block_refile_path = '/home/jupyter-austin2/pc_dataset/ME_dataset/soldier/ply_block/val/ground/'

j = 0
filelist = os.listdir(gt_block_refile_path)
for file in os.listdir(noisy_block_file_path):
    if file not in filelist:
        filelist.append(file)
        j = j + 1
        print(file + ' kd tree search in ground truth: ' + str(j) + '/' + str(len(os.listdir(noisy_block_file_path))))

        noisy_block_file = os.path.join(noisy_block_file_path, file)

        gt_file_name = '_'.join(file.split('_')[:-1]) + '.' + file.split('.')[-1]
        gt_file = os.path.join(gt_file_path, gt_file_name)

        gt_block_refile = os.path.join(gt_block_refile_path, file)

        with open(noisy_block_file, 'rb') as fn:
            plydata = PlyData.read(fn)
            length = len(plydata.elements[0].data)
            data = plydata.elements[0].data
            data_pd = pd.DataFrame(data)
            noisy_block_np = np.zeros(data_pd.shape, dtype=np.float64)
            property_names = data[0].dtype.names

            for i, name in enumerate(property_names):
                noisy_block_np[:, i] = data_pd[name]


        with open(gt_file, 'rb') as fg:
            plydata = PlyData.read(fg)
            length = len(plydata.elements[0].data)
            data = plydata.elements[0].data
            data_pd = pd.DataFrame(data)
            gt_np = np.zeros(data_pd.shape, dtype=np.float64)
            property_names = data[0].dtype.names

            for i, name in enumerate(property_names):
                gt_np[:, i] = data_pd[name]

        gt_block_np = k_nearest_neighbors(noisy_block_np, gt_np, 1)
        with open(gt_block_refile, 'w') as fw:
            fw.write('ply\n')
            fw.write('format ascii 1.0\n')
            fw.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
            fw.write('comment frame_to_world_scale 0.181731\n')
            fw.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
            fw.write('comment width 1023\n')
            fw.write('element vertex ' + str(gt_block_np.shape[0]) + '\n')
            fw.write('property float x\n')
            fw.write('property float y\n')
            fw.write('property float z\n')
            fw.write('property uchar red\n')
            fw.write('property uchar green\n')
            fw.write('property uchar blue\n')
            fw.write('end_header\n')
            for i in range(gt_block_np.shape[0]):
                x = gt_block_np[i][0][0]
                y = gt_block_np[i][0][1]
                z = gt_block_np[i][0][2]
                r = int(gt_block_np[i][0][3])
                g = int(gt_block_np[i][0][4])
                b = int(gt_block_np[i][0][5])
                line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
                fw.write(line + '\n')
    else:
        j += 1
    

