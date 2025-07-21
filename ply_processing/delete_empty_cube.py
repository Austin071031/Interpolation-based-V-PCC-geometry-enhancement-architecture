from plyfile import *
import numpy as np
import pandas as pd
import os
import shutil

cube_path = '/home/jupyter-austin2/HY_dataset/data_202505/r3/cube50/test_dataset/'
save_path = '/home/jupyter-austin2/HY_dataset/data_202505/r3/cube70_nosmall/test_dataset/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
count = 0
for file in os.listdir(cube_path):
    cube_file = os.path.join(cube_path, file)
    with open(cube_file, 'rb') as fb:
        in_plydata = PlyData.read(fb)
        in_length = len(in_plydata.elements[0].data)
        if in_length > 3 :
            print("{} cube is non-empty".format(file))
            shutil.copy(cube_file, save_path)
#             data = in_plydata.elements[0].data
#             cube_pd = pd.DataFrame(data)
#             cube_np = np.zeros(cube_pd.shape, dtype=np.float64)
#             property_names = data[0].dtype.names
            
#             for i, name in enumerate(property_names):
#                  cube_np[:, i] = cube_pd[name]
                    
#             save_file = os.path.join(save_path, file)
#             with open(save_file, 'w') as fw:
#                 fw.write('ply\n')
#                 fw.write('format ascii 1.0\n')
#                 fw.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
#                 fw.write('comment frame_to_world_scale 0.181731\n')
#                 fw.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
#                 fw.write('comment width 1023\n')
#                 fw.write('element vertex ' + str(len(cube_np)) + '\n')
#                 fw.write('property float x\n')
#                 fw.write('property float y\n')
#                 fw.write('property float z\n')
#                 fw.write('property uchar red\n')
#                 fw.write('property uchar green\n')
#                 fw.write('property uchar blue\n')
#                 fw.write('end_header\n')

#                 for i in range(len(cube_np)):
#                     x_n = cube_np[i][0]
#                     y_n = cube_np[i][1]
#                     z_n = cube_np[i][2]
#                     r = int(cube_np[i][3])
#                     g = int(cube_np[i][4])
#                     b = int(cube_np[i][5])
#                     line = str(x_n) + ' ' + str(y_n) + ' ' + str(z_n) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
#                     fw.write(line + '\n')

        else:
            count += 1
         
print("{} files with less size than 400 in total {} files".format(count, len(os.listdir(cube_path))))
            

    
