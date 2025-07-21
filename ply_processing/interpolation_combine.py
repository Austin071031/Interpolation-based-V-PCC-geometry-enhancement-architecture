from plyfile import *
import numpy as np
import pandas as pd
import os

interpolate_path = '/home/jupyter-austin2/HY_dataset/interpolate_50_block_int/train/test_combine/'
noisy_path = '/home/jupyter-austin2/HY_dataset/train/compress/'
combine_path = '/home/jupyter-austin2/HY_dataset/train/test_combine_interpolation/'


# for iters in os.listdir(interpolate_path):
#     print('='*20 + str(iters))
#     iter_in_file = os.path.join(interpolate_path, iters)
#     iter_noisy_file = os.path.join(noisy_path, iters)
#     iter_combine_file = os.path.join(combine_path, iters)
    
for file in os.listdir(interpolate_path):
    print(file)
    interpolate_file = os.path.join(interpolate_path, file)
    noisy_file = os.path.join(noisy_path, file) 
    combine_file = os.path.join(combine_path, file)

    with open(interpolate_file, 'rb') as fa:
        plydata = PlyData.read(fa)
        in_data = plydata.elements[0].data
        in_data_pd = pd.DataFrame(in_data)
        in_data_np = np.zeros(in_data_pd.shape, dtype=np.float64)
        property_names = in_data[0].dtype.names

        for i, name in enumerate(property_names):
            in_data_np[:, i] = in_data_pd[name]

    with open(noisy_file, 'rb') as fb:
        plydata = PlyData.read(fb)
        noisy_data = plydata.elements[0].data
        noisy_data_pd = pd.DataFrame(noisy_data)
        noisy_data_np = np.zeros((noisy_data_pd.shape[0],6), dtype=np.float64)
        property_names = noisy_data[0].dtype.names

        for i, name in enumerate(property_names):
            noisy_data_np[:, i] = noisy_data_pd[name]

    combine_data = np.concatenate((in_data_np, noisy_data_np), axis=0)


    if not os.path.exists(combine_path):
        os.makedirs(combine_path)

    with open(combine_file, 'w') as fw:
        fw.write('ply\n')
        fw.write('format ascii 1.0\n')
        fw.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
        fw.write('comment frame_to_world_scale 0.181731\n')
        fw.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
        fw.write('comment width 1023\n')
        fw.write('element vertex ' + str(len(combine_data)) +'\n')
        fw.write('property float x\n')
        fw.write('property float y\n')
        fw.write('property float z\n')
        fw.write('property uchar red\n')
        fw.write('property uchar green\n')
        fw.write('property uchar blue\n')
        fw.write('end_header\n')
        for i in range(len(combine_data)):
            x = combine_data[i][0]
            y = combine_data[i][1]
            z = combine_data[i][2]
#             r = y_re[i][0]
#             g = u_re[i][0]
#             b = v_re[i][0]
            r = int(combine_data[i][3])
            g = int(combine_data[i][4])
            b = int(combine_data[i][5])
            line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
            fw.write(line + '\n')