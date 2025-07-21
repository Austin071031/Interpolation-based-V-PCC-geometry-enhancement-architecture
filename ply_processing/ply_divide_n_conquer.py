from plyfile import *
import numpy as np
import pandas as pd
import os
### longdress, loot, redandblack, soldier
### noisyinput, ground
### extra_train, train
path = '/home/jupyter-austin2/pc_dataset/MIR_Dataset/ply/qp42_avgunique/val/noisyinput_lost_aligned/'
outpath = '/home/jupyter-austin2/pc_dataset/MIR_Dataset/ply_block42_512/val/noisyinput/'

# sequence_name = 'redandblack_vox10_'
# sequence_name = 'soldier_vox10_0'
# sequence_name = 'longdress_vox10_'
# sequence_name = 'loot_vox10_'
postfix = '_qp42.ply'

frame_start = 1000
frame_end = 1000

if not os.path.exists(outpath):
    os.makedirs(outpath)
    
# for frame in range(frame_start, frame_end+1):
for file in os.listdir(path):
#     file = "wpcoutput_vox10_" + str(frame) + postfix
    print("divide ply into blocks: " + file)
    file_path = os.path.join(path, file)
    outfilename = file.split('.')[0]
    
    block_num = 0
    block_size = 512
    start = 0
    point_size = 0

    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)
        length = len(plydata.elements[0].data)
        data = plydata.elements[0].data
        data_pd = pd.DataFrame(data)
        data_np = np.zeros(data_pd.shape, dtype=np.float64)
        property_names = data[0].dtype.names

        for i, name in enumerate(property_names):
            data_np[:, i] = data_pd[name]

        while (start + block_size) <= length:
            outfile_path = outpath + outfilename + '_' + str(block_num) + '.ply'
            if (start + block_size + block_size) > length:
                point_size = length - start
            else:
                point_size = block_size
            
            with open(outfile_path, 'w') as f:
                f.write('ply\n')
                f.write('format ascii 1.0\n')
                f.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
                f.write('comment frame_to_world_scale 0.181731\n')
                f.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
                f.write('comment width 1023\n')
                f.write('element vertex ' + str(point_size) + '\n')
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
                f.write('end_header\n')
                for i in range(start, start+point_size):
                    x = data_np[i][0]
                    y = data_np[i][1]
                    z = data_np[i][2]
                    r = int(data_np[i][3])
                    g = int(data_np[i][4])
                    b = int(data_np[i][5])
                    line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
                    f.write(line + '\n')
                
            start = start + point_size                
            block_num = block_num + 1
            

#         outfile_path = outpath + outfilename + '_' + str(block_num) + '.ply'
#         with open(outfile_path, 'w') as f:
#             left = length - start
#             f.write('ply\n')
#             f.write('format ascii 1.0\n')
#             f.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
#             f.write('comment frame_to_world_scale 0.181731\n')
#             f.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
#             f.write('comment width 1023\n')
#             f.write('element vertex ' + str(left) + '\n')
#             f.write('property float x\n')
#             f.write('property float y\n')
#             f.write('property float z\n')
#             f.write('property uchar red\n')
#             f.write('property uchar green\n')
#             f.write('property uchar blue\n')
#             f.write('end_header\n')
#             for i in range(start, length):
#                 x = data_np[i][0]
#                 y = data_np[i][1]
#                 z = data_np[i][2]
#                 r = int(data_np[i][3])
#                 g = int(data_np[i][4])
#                 b = int(data_np[i][5])
#                 line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
#                 f.write(line + '\n')
      
print('finished.')
