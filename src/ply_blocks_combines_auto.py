from plyfile import *
import numpy as np
import pandas as pd
import os

blocks_path = '/home/jupyter-austin2/HY_dataset/interpolate_50_block_int/train/test/'
rec_path = '/home/jupyter-austin2/HY_dataset/interpolate_50_block_int/train/test_combine/'

if not os.path.exists(rec_path):
    os.makedirs(rec_path)

filelist = os.listdir(blocks_path)
filelist.sort(key=lambda x : int(x.split('.')[0].split('_')[-2]))

rec_pc = []

current_frame = filelist[0].split('_')[-2]
start = 0
end = 0
tot_length = 0
for i in range (len(filelist) + 1):
    # check if we meet next frame or reach last frame, yes --> reconstruct the current frame
    if(i == len(filelist) or (filelist[i].split('_')[-2] != current_frame)):
        print('Reconstruct Frame: ' + str(filelist[i-1].split('_')[-3]))
        end = i - 1
        suffix = filelist[end].split('.')[-1]
        filename = "_".join(filelist[end].split('_')[:-1])
        rec_file = rec_path + filename + '.' + suffix
        with open(rec_file, 'w') as fw:
            for j in range(start, end):
                # write header for point cloud
                if(j == start):
                    fw.write('ply\n')
                    fw.write('format ascii 1.0\n')
                    fw.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
                    fw.write('comment frame_to_world_scale 0.181731\n')
                    fw.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
                    fw.write('comment width 1023\n')
                    fw.write('element vertex ' + str(tot_length) +'\n')
                    fw.write('property float x\n')
                    fw.write('property float y\n')
                    fw.write('property float z\n')
                    fw.write('property uchar red\n')
                    fw.write('property uchar green\n')
                    fw.write('property uchar blue\n')
                    fw.write('end_header\n')

                #write data for point cloud    
                file_path = os.path.join(blocks_path, filelist[j])
                with open(file_path, 'rb') as f:
                    plydata = PlyData.read(f)
                    length = len(plydata.elements[0].data)
                    data = plydata.elements[0].data
                    data_pd = pd.DataFrame(data)
                    data_np = np.zeros(data_pd.shape, dtype=np.float64)
                    property_names = data[0].dtype.names

                    for y, name in enumerate(property_names):
                        data_np[:, y] = data_pd[name]

                    f.close()

                for h in range(len(data_np)):
                        x = data_np[h][0]
                        y = data_np[h][1]
                        z = data_np[h][2]
                        r = int(data_np[h][3])
                        g = int(data_np[h][4])
                        b = int(data_np[h][5])
                        line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
                        fw.write(line + '\n')

            fw.close()

        if(end + 1 != len(filelist)):
            # update index, current frame and total lines 
            start = i
            current_frame = filelist[start].split('_')[-2]
            tot_length = 0

    if(i != len(filelist)):
        file_path = os.path.join(blocks_path, filelist[i])
        with open(file_path, 'rb') as f1:
            plydata = PlyData.read(f1)
            length = len(plydata.elements[0].data)
            data = plydata.elements[0].data
            data_pd = pd.DataFrame(data)

            # sum up total lines for each point cloud
            tot_length += data_pd.shape[0]

            f1.close()
