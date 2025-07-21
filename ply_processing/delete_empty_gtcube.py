from plyfile import *
import numpy as np
import pandas as pd
import os
import shutil

cube_path = '/home/jupyter-austin2/pc_dataset/MIR_Dataset/ply_cube/ply_qp36_cube40_overlap10/train/noisyinput_nosmoll/'
gt_path = '/home/jupyter-austin2/pc_dataset/MIR_Dataset/ply_cube/ply_qp36_cube40_overlap10/train/ground/'
cube_emptyinGT_path = '/home/jupyter-austin2/pc_dataset/MIR_Dataset/ply_cube/ply_qp36_cube40_overlap10/train/noisyinput_emptyinGT/'

if not os.path.exists(cube_emptyinGT_path):
    os.makedirs(cube_emptyinGT_path)

no_count = 0
for file in os.listdir(cube_path):
    cube_file = os.path.join(cube_path, file)
    gt_file = os.path.join(gt_path, file)
    cube_emptyinGT_file = os.path.join(cube_emptyinGT_path, file)
    with open(gt_file, 'rb') as fb:
        in_plydata = PlyData.read(fb)
        in_length = len(in_plydata.elements[0].data)
        if in_length == 0:
            no_count += 1
            shutil.move(cube_file, cube_emptyinGT_file)
            
else:
    print("these are {} interpolated file has no points in aligned ground patch".format(no_count))
    print("finished")

        
            

    
