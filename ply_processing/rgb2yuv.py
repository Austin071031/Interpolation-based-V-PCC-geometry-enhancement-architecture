from plyfile import *
import numpy as np
import pandas as pd
import os
#/pc_dataset/ME_dataset_multiqp/QP37/coord_optimized_unique_align_dataset/longdress/train/noisyinp#ut
blocks_path = '/home/jupyter-austin2/HY_dataset/block/data/test_dataset/compress/block_combine/interpolate_20/'
rec_path = '/home/jupyter-austin2/HY_dataset/block/data/test_dataset/compress/block_combine/interpolate_20re/'

sequence_name = 'soldier_vox10_0'
# sequence_name = 'redandblack_vox10_'
# sequence_name = 'longdress_vox10_'
# sequence_name = 'loot_vox10_'
postfix = '.ply'

frame_start = 536
frame_end = 550

# if not os.path.exists(rec_path):
#     os.makedirs(rec_path)


# for i in range(frame_start, frame_end+1):
# for iters in os.listdir(blocks_path):
#     file_path = os.path.join(blocks_path, iters)
#     out_path = os.path.join(rec_path, iters)
if not os.path.exists(out_path):
    os.makedirs(out_path)

for file in os.listdir(blocks_path):
#     file = sequence_name + str(i) + postfix
    print("rgb to yuv:" + file)
    pc_path = os.path.join(blocks_path, file)
    rec_file_path = os.path.join(rec_path, file)

    with open(pc_path, 'rb') as f:
         plydata = PlyData.read(f)
         length = len(plydata.elements[0].data)
         data = plydata.elements[0].data
         data_pd = pd.DataFrame(data)
         data_np = np.zeros(data_pd.shape, dtype=np.float64)
         property_names = data[0].dtype.names

         for i, name in enumerate(property_names):
             data_np[:, i] = data_pd[name]


#         y = 0.2126*data_np[:, 3] + 0.7152*data_np[:, 4] + 0.0722*data_np[:, 5]
#         y_re = y.reshape(-1, 1)

#         u = -0.1146*data_np[:, 3] - 0.3854*data_np[:, 4] + 0.5000*data_np[:, 5]
#         u_re = u.reshape(-1, 1)

#         v = 0.5000*data_np[:, 3] - 0.4542*data_np[:, 4] - 0.0458*data_np[:, 5]
#         v_re = v.reshape(-1, 1)



    with open(rec_file_path, 'w') as fw:
        fw.write('ply\n')
        fw.write('format ascii 1.0\n')
        fw.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
        fw.write('comment frame_to_world_scale 0.181731\n')
        fw.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
        fw.write('comment width 1023\n')
        fw.write('element vertex ' + str(len(data_np)) +'\n')
        fw.write('property float x\n')
        fw.write('property float y\n')
        fw.write('property float z\n')
        fw.write('property uchar red\n')
        fw.write('property uchar green\n')
        fw.write('property uchar blue\n')
        fw.write('end_header\n')
        for i in range(len(data_np)):
            x = data_np[i][0]
            y = data_np[i][1]
            z = data_np[i][2]
#             r = y_re[i][0]
#             g = u_re[i][0]
#             b = v_re[i][0]
            r = int(data_np[i][3])
            g = int(data_np[i][4])
            b = int(data_np[i][5])
            line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
            fw.write(line + '\n')

    fw.close()
