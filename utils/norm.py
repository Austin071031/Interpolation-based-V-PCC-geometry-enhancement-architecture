import torch
import numpy as np
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler


def mean_n_std(data):
    """ data: point cloud data in shape (N, 3) """
    mean = torch.zeros(1)
    std = torch.zeros(1)
    max_value = torch.zeros(1)
    
    mean = data.mean(1).sum(0)
    std = data.std(1).sum(0)
    max_value = data.max()
    
    return mean, std, max_value

def normalize(data):
    """ data: point cloud coordinates in shape(N, 3) """
    max_value = data.max()
    
    # scale xyz value to 0 ~ 1
    tensor_data = torch.from_numpy(data)
    tensor_data = tensor_data / max_value
    
#     mean = torch.mean(tensor_data, dim=0)
#     std = torch.std(tensor_data, dim=0)
    tensor_data = tensor_data.reshape(1, -1, data.shape[-1])
    # Normalize
    norm_data = transforms.Normalize(0.5, 1)(tensor_data)
    
    return norm_data, max_value

def unnormalize(data, mean, std, max_value):
    """ data: normalized point cloud coordinates in shape(N, 3) """
#     data_re = data.reshape(data.shape[-1], -1, 1)
#     print("-----------reshape normalized data-----------")
#     print(data_re)
    
    data_unnorm = data * std + mean
    data_unnorm = data_unnorm * max_value
    
    return data_unnorm.reshape(-1, data.shape[-1])


def min_max_normalize(data, feature_range):
    """ 
    data: point cloud coordinates in numpy array, shape(N, 3) 
    feature_range: scale data into range (0, feature_range)
    """
    min_max_scaler = MinMaxScaler(feature_range=(0, feature_range))
    min_max_scaler = min_max_scaler.fit(data)
    norm_data = min_max_scaler.transform(data)
    
    return min_max_scaler, norm_data

def min_max_unnormalize(min_max_scaler, norm_data, noisyinput):
    """ 
    min_max_scaler: MinMaxScaler from normalize process
    data: point cloud coordinates in numpy array, shape(N, 3) 
    """
    scalar = norm_data[:,0]
    scalar = scalar.unsqueeze(1)
    vector = norm_data[:,1:]
    _,targets = vector.max(dim=1)
    t = targets.unsqueeze(1)
    onehot = torch.zeros(vector.shape).cuda()
    onehot.scatter_(1, t, 1)
    proj = onehot*scalar


    output = noisyinput+proj
    
    rec_data = min_max_scaler.inverse_transform(output.cpu())
    
    return rec_data


# data = np.array([[1.5, 2, 3],
#         [2, 3, 4.2],
#         [3.5, 4, 5]])

# min_max_scaler, norm_data = min_max_normalize(data, 10)
# print(norm_data)
# unnorm_data = min_max_unnormalize(min_max_scaler, norm_data)
# print(unnorm_data)