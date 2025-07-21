# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:15:13 2019

@author: Anique

run prepare_data_2.py before running this file.
"""
import os, glob
import numpy as np
## import open3d as opn3d
from datetime import datetime
from plyfile import *
import pandas as pd

# Variables to be changed
save_dir = "/home/jupyter-austin2/pc_dataset/ME_dataset/soldier/ply/"
rate = ['rgb']

### Both Set
t1 = datetime.now()
for r in rate:
    print('==============',r,'===============')
    for k in ['train','val']:
        print('======!!!=====',k,'=======!!!=====')
        files = sorted(glob.glob(save_dir+ r+'/'+k+'/noisyinput/*.ply'))
        
        groundtruth = []
        noisyinput = []
        i = 0
        for f in files:
            i += 1
            print(i, "/", len(files))
            
            f2 = f.replace('noisyinput','ground')
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
            colors_noisy[:, 0] = data_pd['red']
            colors_noisy[:, 1] = data_pd['green']
            colors_noisy[:, 2] = data_pd['blue']                

            with open(f2, 'rb') as f2:
                plydata = PlyData.read(f2)
                length = len(plydata.elements[0].data)
                data = plydata.elements[0].data
                data_pd = pd.DataFrame(data)
                coords_gt = np.zeros((data_pd.shape[0], 3), dtype=np.float32)
                colors_gt = np.zeros((data_pd.shape[0], 3), dtype=np.float32)                
                property_names = data[0].dtype.names

            coords_gt[:, 0] = data_pd['x']
            coords_gt[:, 1] = data_pd['y']
            coords_gt[:, 2] = data_pd['z']
            colors_gt[:, 0] = data_pd['red']
            colors_gt[:, 1] = data_pd['green']
            colors_gt[:, 2] = data_pd['blue']                

            xyz = np.asarray(coords_noisy)
            rgb = np.asarray(colors_noisy)
            xyz2 = np.asarray(coords_gt) 
            rgb2 = np.asarray(colors_gt)
            
            if xyz.shape[0]==0 or xyz2.shape[0]==0:
                print("File NOT Copied !!")
                continue
            
            ground = np.concatenate((xyz2,rgb2),axis=1)
            noisy = np.concatenate((xyz,rgb), axis=1)
            groundtruth.append(ground)
            noisyinput.append(noisy)
            print("File Copied !!")
        
        file_name = "pointsets_"+k+".npz"
        np.savez(os.path.join(save_dir, r, file_name), groundtruth=groundtruth, noisyinput=noisyinput)

t2 = datetime.now()
print(str(t2-t1))
