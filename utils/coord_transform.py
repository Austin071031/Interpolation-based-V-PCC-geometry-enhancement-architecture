import torch

def transform_relative(data):
    """
    data: point cloud coords data, torch.tensor, (N, 3)
    """
    center = data[0, :]
    relative_data = torch.zeros(data.shape)
    relative_data[1:, :] = data[1:, :] - center
    
    return relative_data, center


def inverse_transform_relative(data, center):
    """
    data: point cloud coords data, torch.tensor, (N, 3)
    center: center point of data, torch.tensor, (1, 3)
    """
    inverse_data = torch.zeros(data.shape)
    inverse_data[:, 0] = data[:, 0]
    inverse_data[:, 1:] = data[:, 1:] + center
    
    return inverse_data