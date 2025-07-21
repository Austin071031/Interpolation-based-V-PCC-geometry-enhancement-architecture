import torch
import numpy as np
from sklearn.neighbors import KDTree



def create_dic_fuse(noisyinput, combinepc):
    copy = noisyinput

#### build dictionary based on noisyinput
    coords = noisyinput[:, :3]
    colors_noisy = noisyinput[:, 3:]
    
    colors_yuv = np.zeros(colors_noisy.shape)
    colors_yuv[:, 0] = 0.2126*colors_noisy[:, 0] + 0.7152*colors_noisy[:, 1] + 0.0722*colors_noisy[:, 2]

    colors_yuv[:, 1] = -0.1146*colors_noisy[:, 0] - 0.3854*colors_noisy[:, 1] + 0.5000*colors_noisy[:, 2]

    colors_yuv[:, 2] = 0.5000*colors_noisy[:, 0] - 0.4542*colors_noisy[:, 1] - 0.0458*colors_noisy[:, 2]

    # add one more column to colors to count how many patches are inclued the same point 
    count = np.zeros(colors_noisy.shape[0])
    colors_cnt = np.insert(colors_yuv, 0, count, axis=1)

    noisyinput_dic = {}
    for o in range(len(coords)):
        key = " ".join(str(x) for x in coords[o])
        value = colors_cnt[o]
        noisyinput_dic[key] = value

#### fuse combine pc, take average to those point included in multiple patches.
    for p in range(len(combinepc[:])):
        coords = " ".join(str(x) for x in combinepc[p, :3])
        colors_processed = combinepc[p, 3:]

        temp = noisyinput_dic.get(coords)
        updated_value = np.ones(temp.shape)
        if temp[0] == 0:
            updated_value[0] = temp[0] + 1
            updated_value[1] = colors_processed[0]
            updated_value[2] = colors_processed[1]
            updated_value[3] = colors_processed[2]
        else:
            updated_value[0] = temp[0] + 1
            updated_value[1] = temp[1] + colors_processed[0]
            updated_value[2] = temp[2] + colors_processed[1]
            updated_value[3] = temp[3] + colors_processed[2]

        noisyinput_dic[coords] = updated_value

#### convert fused pc back to array
    k = 0
    for key, values in noisyinput_dic.items():
        point = np.zeros(6)
        coords = key.split(" ")
        colors_fused = np.zeros(3)

        if values[0] == 0:
            colors_fused[0] = values[1]
            colors_fused[1] = values[2]
            colors_fused[2] = values[3]
        else:
            colors_fused[0] = values[1] / values[0]
            colors_fused[1] = values[2] / values[0]
            colors_fused[2] = values[3] / values[0]

        point[0] = coords[0]
        point[1] = coords[1]
        point[2] = coords[2]
        point[3] = colors_fused[0]
        point[4] = colors_fused[1]
        point[5] = colors_fused[2]

        copy[k] = point
        k += 1

    return copy

def get_metrics(keep, target):
    with torch.no_grad():
        TP = (keep * target).nonzero().shape[0]
        FN = (~keep * target).nonzero().shape[0]
        FP = (keep * ~target).nonzero().shape[0]
        # TN = (~keep * ~target).nonzero().shape[0]

        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        IoU = TP / (TP + FP + FN + 1e-7)

    return [round(precision, 4), round(recall, 4), round(IoU, 4)]

import torch
import numpy as np
from sklearn.neighbors import KDTree

def k_nearest_neighbors(A, B, K):
    # convert numpy arrays to PyTorch tensors
#     A = torch.from_numpy(A).float()
#     B = torch.from_numpy(B).float()

    # split the position and color information
    A_pos, A_col = A[:, :3], A[:, 3:]
    B_pos, B_col = B[:, :3], B[:, 3:]
    

    # build a KD-Tree from the positions of point cloud B
    tree = KDTree(B_pos.numpy())

    # find the indices of the K nearest neighbors for each point in A
    distances, indices = tree.query(A_pos.numpy(), k=K)

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
        nn = torch.cat((nn_pos, nn_col), dim=-1)

        # store the result for this point
        result[i] = nn.numpy()

    return result

import torch
import torch.nn.functional as F

def point_cloud_color_loss(pred_cloud, target_cloud):
    """
    自定义点云颜色特征差异损失函数
    :param pred_cloud: 预测点云，形状为 (B, N, 3)，其中 B 表示批次大小，N 表示点的数量
    :param target_cloud: 目标点云，形状为 (B, N, 3)，其中 B 表示批次大小，N 表示点的数量
    :return: 损失值
    """
    # 将点云颜色特征拆分出来
    pred_colors = pred_cloud[:, :, 3:]
    target_colors = target_cloud[:, :, 3:]

    # 计算颜色特征差异
    color_diff = (pred_colors - target_colors).abs()

    # 对每个点的颜色特征差异求平均，并对整个点云求和
    color_diff = color_diff.mean(dim=2).sum(dim=1)

    # 对损失值进行归一化处理
    loss = F.normalize(color_diff, p=2, dim=0)

    return loss

