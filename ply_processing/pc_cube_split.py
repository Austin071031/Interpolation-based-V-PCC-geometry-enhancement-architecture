from plyfile import *
import numpy as np
import pandas as pd
import os, time

# train, val, noisyinput, ground_lost, ground_align

pc_file = "/home/jupyter-austin2/HY_dataset/data_202505/r3/ply/test_dataset/"
save_cube_file = "/home/jupyter-austin2/HY_dataset/data_202505/r3/cube50/test_dataset/"

cube_size = 50



postfix = '.ply'



if not os.path.exists(save_cube_file):
    os.makedirs(save_cube_file)

for file in os.listdir(pc_file):
    print("------File cube split: " + file)
    pc_file_path = os.path.join(pc_file, file)
    
    with open(pc_file_path, 'rb') as f:
        plydata = PlyData.read(f)
        vertex = plydata['vertex']
#         aligned_np = np.vstack([vertex['x'], vertex['y'], vertex['z'], vertex['red'],vertex['green'],vertex['blue']]).T
        aligned_np = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
#         length = len(plydata.elements[0].data)
#         data = plydata.elements[0].data
#         aligned_pd = pd.DataFrame(data)
#         aligned_np = np.zeros(aligned_pd.shape, dtype=np.float64)
#         property_names = data[0].dtype.names


#         for i, name in enumerate(property_names):
#              aligned_np[:, i] = aligned_pd[name]
            
    
#     max_x = max(aligned_np[:,0])
#     max_y = max(aligned_np[:,1])
#     max_z = max(aligned_np[:,2])
    # 计算切割后的小正方体数量
    num_patches = (1024 // cube_size + 1) * (1024 // cube_size + 1) * (1024 // cube_size + 1)
    print("------number of cube patches: ", num_patches)
    # 遍历每个小正方体
    s = time.time()
    count = 0
    max_point_cube = 0
    min_point_cube = 0
    for x in range(0, 1024, cube_size):
        for y in range(0, 1024, cube_size):
            for z in range(0, 1024, cube_size):
                if z + cube_size > 1024:
                    z = 1024 - cube_size
                if y + cube_size > 1024:
                    y = 1024 - cube_size
                if x + cube_size > 1024:
                    x = 1024 - cube_size
                # 获取当前小正方体内的所有点
                patch_points = aligned_np[(aligned_np[:, 0] >= x) & (aligned_np[:, 0] < x + cube_size) &
                                      (aligned_np[:, 1] >= y) & (aligned_np[:, 1] < y + cube_size) &
                                      (aligned_np[:, 2] >= z) & (aligned_np[:, 2] < z + cube_size)]
#                 if(len(patch_points) != 0):
                cube_file_name = file.split('.')[0] + "_" + str(count) + ".ply"
                save_cube_file_path = os.path.join(save_cube_file, cube_file_name)
                with open(save_cube_file_path, 'w') as fw:
                    fw.write('ply\n')
                    fw.write('format ascii 1.0\n')
                    fw.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
                    fw.write('comment frame_to_world_scale 0.181731\n')
                    fw.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
                    fw.write('comment width 1023\n')
                    fw.write('element vertex ' + str(len(patch_points)) + '\n')
                    fw.write('property float x\n')
                    fw.write('property float y\n')
                    fw.write('property float z\n')
#                     fw.write('property uchar red\n')
#                     fw.write('property uchar green\n')
#                     fw.write('property uchar blue\n')
                    fw.write('end_header\n')
                    
                    if(len(patch_points) != 0):
                        for i in range(len(patch_points)):
                            x_n = patch_points[i][0]
                            y_n = patch_points[i][1]
                            z_n = patch_points[i][2]
#                             r = int(patch_points[i][3])
#                             g = int(patch_points[i][4])
#                             b = int(patch_points[i][5])
#                             line = str(x_n) + ' ' + str(y_n) + ' ' + str(z_n) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
                            line = str(x_n) + ' ' + str(y_n) + ' ' + str(z_n)
                            fw.write(line + '\n')
                if len(patch_points) > max_point_cube:
                    max_point_cube = len(patch_points)
                if len(patch_points) < min_point_cube or min_point_cube == 0:
                    min_point_cube = len(patch_points)
                count = count + 1
    print("count: {}".format(count))      
    print("max point in cube: {}".format(max_point_cube))
    print("min point in cube: {}".format(min_point_cube))
    print("cube split time for one ply file: ", time.time() - s)
print("-------------finished-------------")
                
        