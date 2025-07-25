import open3d as o3d
import numpy as np
import pandas as pd
import h5py
import torch
import MinkowskiEngine as ME
import os
import time
from plyfile import *
from torch.utils.data.sampler import Sampler


def loadh5(filedir, color_format='rgb'):
  """Load coords & feats from h5 file.

  Arguments: file direction

  Returns: coords & feats.
  """
  pc = h5py.File(filedir, 'r')['data'][:]

  coords = pc[:,0:3].astype('int32')

  if color_format == 'rgb':
    feats = pc[:,3:6]/255. 
  elif color_format == 'yuv':
    R, G, B = pc[:, 3:4], pc[:, 4:5], pc[:, 5:6]
    Y = 0.257*R + 0.504*G + 0.098*B + 16
    Cb = -0.148*R - 0.291*G + 0.439*B + 128
    Cr = 0.439*R - 0.368*G - 0.071*B + 128
    feats = np.concatenate((Y,Cb,Cr), -1)/256.
  elif color_format == 'geometry':
    feats = np.expand_dims(np.ones(coords.shape[0]), 1)
  elif color_format == 'None':
    return coords
    
  feats = feats.astype('float32')

  return coords, feats

def loadply(filedir, color_format='geometry'):
  """Load coords & feats from ply file.
  
  Arguments: file direction.
  
  Returns: coords & feats.
  """
#   pcd = o3d.io.read_point_cloud(filedir)
#   coords = np.asarray(pcd.points)
#   feats = np.asarray(pcd.colors)

#   coords = np.asarray(pcd.colors)
  with open(filedir, 'rb') as f:
    plydata = PlyData.read(f)
    length = len(plydata.elements[0].data)
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)

    coords_np = np.zeros((data_pd.shape[0], 3), dtype=np.float64)    
    colors_np = np.zeros((data_pd.shape[0], 3), dtype=np.float64)
    property_names = data[0].dtype.names


    coords_np[:, 0] = data_pd['x']
    coords_np[:, 1] = data_pd['y']
    coords_np[:, 2] = data_pd['z']

    colors_np[:, 0] = data_pd['red']
    colors_np[:, 1] = data_pd['green']
    colors_np[:, 2] = data_pd['blue']

  coords = coords_np
    
  if color_format=='geometry':
#     feats = np.expand_dims(np.ones(coords.shape[0]), 1)
#     feats = np.asarray(pcd.colors)
    feats = colors_np
  elif color_format == 'None':
    feats = colors_np
    return coords, feats
  
  feats = feats.astype('float32')
  return coords, feats

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)



class Dataset(torch.utils.data.Dataset):

    def __init__(self, files, GT_folder, downsample, feature_format='geometry'):
        self.coords = []
        self.feats = []
        self.coords_T = []
        self.feats_T = []
        self.downsample = downsample
        self.filedir = []
        if GT_folder==None:     ## Finding out whether to downsample or not.
            self.ds = True
        else:
            self.ds = False
        
        for i,f in enumerate(files):
            if self.ds:     # If need to downsample
                if f.endswith('.h5'):
                    coords, feats = loadh5(f, feature_format)
                    coords_T = coords
                elif f.endswith('.ply'):
                    coords, feats = loadply(f, feature_format)
                    coords_T = coords
                self.filedir.append(f)
            else:
                self.filedir.append(f)
                name = os.path.basename(f)
                gt_file = os.path.join(GT_folder, name)
                if not os.path.exists(gt_file):
                    print(gt_file)
                    print('Error, File does not exist in GT folder')
                    continue
                
                if f.endswith('.h5'):
                    coords, feats = loadh5(f, feature_format)
                    coords_T = loadh5(gt_file, 'None')
                elif f.endswith('.ply'):
                    coords, feats = loadply(f, feature_format)
                    coords_T, feats_T = loadply(gt_file, 'None')
                    
                    
            
            
            self.coords.append(coords)
            self.feats.append(feats)
            self.coords_T.append(coords_T)
            self.feats_T.append(feats_T)
            
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.ds:
            coords = self.coords[idx]
            feats = self.feats[idx]
            coords_T = self.coords_T[idx]
            feats_T = self.feats_T[idx]
            filedir = self.filedir[idx]
            
            N = coords_T.shape[0]
            N2 = N//self.downsample
            idx = np.random.choice(N, N2, replace=False)
            coords = coords[idx]
            feats = feats[idx]
            return (coords, feats, coords_T, filedir)
        else:
#             print(self.coords[idx].shape)
#             print(str(self.feats[idx].shape) + "noisy")
#             print(self.coords_T[idx].shape)
#             print(str(self.feats_T[idx].shape) + "GT")
            
            return (self.coords[idx], self.feats[idx], self.coords_T[idx], self.feats_T[idx], self.filedir[idx])
        

def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    coords, feats, coords_T, feats_T, filedir = list(zip(*list_data))

    coords_batch = ME.utils.batched_coordinates(coords)
    feats_batch = torch.from_numpy(np.vstack(feats)).float()
#     coords_T_batch = ME.utils.batched_coordinates(coords_T)
    coords_T_batch = ME.utils.batched_coordinates(coords_T)
    feats_T_batch = torch.from_numpy(np.vstack(feats_T)).float()

    return coords_batch, feats_batch, coords_T_batch, feats_T_batch, filedir
    
def collate_test_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    coords, feats, coords_T, filedir = list(zip(*list_data))

    coords_batch = ME.utils.batched_coordinates(coords)
    feats_batch = torch.from_numpy(np.vstack(feats)).float()
#     coords_T_batch = ME.utils.batched_coordinates(coords_T)
    coords_T_batch = torch.from_numpy(np.vstack(coords_T)).float()

    return coords_batch, feats_batch, coords_T_batch, filedir

def make_data_loader(files, GT_folder, batch_size, downsample, shuffle, num_workers, repeat):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_pointcloud_fn, 
        'pin_memory': True,
        'drop_last': False
    }
    
    start_time = time.time()
    print("Going to load the whole dataset in the memory, No. of files = ", len(files))
    dataset = Dataset(files, GT_folder, downsample)
    print("Time taken to load the dataset: ", round(time.time() - start_time, 4))
    
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    
    loader = torch.utils.data.DataLoader(dataset, **args)
    
    return loader
    
def make_test_data_loader(files, GT_folder, batch_size, downsample, shuffle, num_workers, repeat):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_test_pointcloud_fn, 
        'pin_memory': True,
        'drop_last': False
    }
    
    start_time = time.time()
    print("Going to load the whole dataset in the memory, No. of files = ", len(files))
    dataset = Dataset(files, GT_folder, downsample)
    print("Time taken to load the dataset: ", round(time.time() - start_time, 4))
    
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    
    loader = torch.utils.data.DataLoader(dataset, **args)
    
    return loader

