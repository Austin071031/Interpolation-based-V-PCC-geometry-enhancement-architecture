from plyfile import *
import numpy as np
import pandas as pd
import os

blocks_path = '/home/jeff/Research/Point_Cloud/Minkowski/Result/conquer_ply/combine/'
rec_path = '/home/jeff/Research/Point_Cloud/Minkowski/Result/conquer_ply/reconstruct/output/'

filelist = os.listdir(blocks_path)
filelist.sort(key=lambda x : int(x.split('.')[0].split('_')[-1]))

rec_pc = []

rec_file = rec_path + '1.ply'

with open(rec_file, 'w') as fw:
    for file in filelist:
        file_path = os.path.join(blocks_path, file)

        with open(file_path, 'rb') as f:
            plydata = PlyData.read(f)
            length = len(plydata.elements[0].data)
            data = plydata.elements[0].data
            data_pd = pd.DataFrame(data)
            data_np = np.zeros(data_pd.shape, dtype=np.float64)
            property_names = data[0].dtype.names

            for i, name in enumerate(property_names):
                data_np[:, i] = data_pd[name]

        for i in range(len(data_np)):
                x = int(data_np[i][0])
                y = int(data_np[i][1])
                z = int(data_np[i][2])
                r = int(data_np[i][3])
                g = int(data_np[i][4])
                b = int(data_np[i][5])
                line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
                fw.write(line + '\n')    

