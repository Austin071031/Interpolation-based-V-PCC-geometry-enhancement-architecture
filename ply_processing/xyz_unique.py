from plyfile import *
import numpy as np
import pandas as pd
import os

### soldier, longdress, loot, redandblack

path = '/home/jupyter-austin2/HY_dataset/train/test_combine_interpolation/'
outfile = '/home/jupyter-austin2/HY_dataset/train/test_combine_interpolation_unique/'

# sequence_name = 'redandblack_vox10_'
sequence_name = 'soldier_vox10_0'
# sequence_name = 'longdress_vox10_'
# sequence_name = 'loot_vox10_'
postfix = '.ply'

frame_start = 536
frame_end = 550

if not os.path.exists(outfile):
    os.makedirs(outfile)

# for frame in range(frame_start, frame_end+1):
for file in os.listdir(path):
#     file = sequence_name + str(frame) + postfix
    print("File unique coordinates: " + file)
    file_path = os.path.join(path, file)
    out_path = os.path.join(outfile, file)

    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)
        length = len(plydata.elements[0].data)
        data = plydata.elements[0].data
        data_pd = pd.DataFrame(data)
        print(data_pd.shape)
#         print(data_pd)
#         data_pd[['x', 'y', 'z']] = data_pd[['x', 'y', 'z']].astype(int) #向下取整
#         data_pd[['x', 'y', 'z']] = data_pd[['x', 'y', 'z']].round() #四舍五入
#         data_pd_unique = data_pd.drop_duplicates(subset=['x', 'y', 'z'])
        data_pd_gp = data_pd.groupby(['x', 'y', 'z']).mean()
        s = pd.Series(range(data_pd_gp.shape[0]), index=data_pd_gp.index)
        data_pd_avg = pd.concat([s.reset_index(), pd.DataFrame(data_pd_gp.values)], axis=1, ignore_index=True)
        data_pd_avg.drop(data_pd_avg.columns[3], axis=1, inplace=True)
        data_pd_avg.rename(columns = {0:'x', 1:'y', 2:'z', 4: 'red', 5: 'green', 6:'blue'}, inplace=True)
        print(data_pd_avg.shape)
#         print(data_pd_avg)
        data_np = np.zeros(data_pd_avg.shape, dtype=np.float64)
        property_names = data[0].dtype.names

        for i, name in enumerate(property_names):
            data_np[:, i] = data_pd_avg[name]
            
#####     xyz unique for floating point of coordinates
#         xyz_int = np.ceil(data_np[:, :3])
#         data_np_int = np.concatenate((xyz_int, data_np[:, 3:]), axis=1)
#         df = pd.DataFrame(data_np_int, columns=['x', 'y', 'z', 'red', 'green', 'blue'])
#         data_pd_unique = df.drop_duplicates(subset=['x', 'y', 'z'], keep=False)
#         print(data_pd_unique.shape)

#         data_np_unique = np.zeros(data_pd_unique.shape, dtype=np.float64)
#         for i, name in enumerate(property_names):
#             data_np_unique[:, i] = data_pd_unique[name]

        with open(out_path, 'w') as fw:
            fw.write('ply\n')
            fw.write('format ascii 1.0\n')
            fw.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
            fw.write('comment frame_to_world_scale 0.181731\n')
            fw.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
            fw.write('comment width 1023\n')
            fw.write('element vertex ' + str(len(data_np[:])) + '\n')
            fw.write('property float x\n')
            fw.write('property float y\n')
            fw.write('property float z\n')
            fw.write('property uchar red\n')
            fw.write('property uchar green\n')
            fw.write('property uchar blue\n')
            fw.write('end_header\n')
            for i in range(len(data_np[:])):
                x_n = data_np[i][0]
                y_n = data_np[i][1]
                z_n = data_np[i][2]
                r = int(data_np[i][3])
                g = int(data_np[i][4])
                b = int(data_np[i][5])
                line = str(x_n) + ' ' + str(y_n) + ' ' + str(z_n) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
                fw.write(line + '\n')
print('finished')