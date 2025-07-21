from plyfile import *
import numpy as np
import pandas as pd
import os

blocks_path = '/home/jupyter-austin2/pc_dataset/QP47_fpssampling/QP47/val/noisyinput/'
noisyinput_path = '/home/jupyter-austin2/pc_dataset/ME_dataset_multiqp/QP47/coord_optimized_unique_align_dataset/'
rec_path = '/home/jupyter-austin2/pc_dataset/QP47_fpssampling/test/'

if not os.path.exists(rec_path):
    os.makedirs(rec_path)

    
def create_dic_fuse(noisyinput, combinepc):
    copy = noisyinput
    
#### build dictionary based on noisyinput
    coords = noisyinput[:, :3]
    colors = noisyinput[:, 3:]
    
    # add one more column to colors to count how many patches are inclued the same point 
    count = np.zeros(colors.shape[0], dtype=np.int)
    colors_cnt = np.insert(colors, 0, count, axis=1)
    
    noisyinput_dic = {}
    for o in range(len(coords)):
        key = " ".join(str(x) for x in coords[o])
        value = colors_cnt[o]
        noisyinput_dic[key] = value

#### fuse combine pc, take average to those point included in multiple patches.
    for p in range(len(combinepc[:])):
        coords = " ".join(str(x) for x in combinepc[p, :3])
        colors = combinepc[p, 3:]
        
        temp = noisyinput_dic.get(coords)
        updated_value = np.ones(temp.shape)
        if temp[0] == 0:
            updated_value[0] = temp[0] + 1
            updated_value[1] = colors[0]
            updated_value[2] = colors[1]
            updated_value[3] = colors[2]
        else:
            updated_value[0] = temp[0] + 1
            updated_value[1] = temp[1] + colors[0]
            updated_value[2] = temp[2] + colors[1]
            updated_value[3] = temp[3] + colors[2]
            
        noisyinput_dic[coords] = updated_value
    
#### convert fused pc back to array
    k = 0
    for key, values in noisyinput_dic.items():
        point = np.zeros(6)
        coords = key.split(" ")
        colors = np.zeros(3)
        
        if values[0] == 0:
            colors[0] = values[1]
            colors[1] = values[2]
            colors[2] = values[3]
        else:     
            colors[0] = values[1] / values[0]
            colors[1] = values[2] / values[0]
            colors[2] = values[3] / values[0]
        point[0] = coords[0]
        point[1] = coords[1]
        point[2] = coords[2]
        point[3] = colors[0]
        point[4] = colors[1]
        point[5] = colors[2]
        
        copy[k] = point
        k += 1
        
    return copy
 
    
filelist = os.listdir(blocks_path)
filelist.sort(key=lambda x : int(x.split('.')[0].split('_')[-3]))

rec_pc = []

current_frame = filelist[0].split('_')[-3]
start = 0
end = 0
tot_length = 0
for i in range (len(filelist) + 1):
    # check if we meet next frame or reach last frame, yes --> reconstruct the current frame
    if(i == len(filelist) or (filelist[i].split('_')[-3] != current_frame)):
        end = i - 1
        suffix = filelist[end].split('.')[-1]
        filename = "_".join(filelist[end].split('_')[:-1])
        sequence_name = filename.split('_')[0]
        rec_file = rec_path + filename + '.' + suffix
        
        ### get noisyinput pc as reference      
        noisyinput_file = os.path.join(noisyinput_path, sequence_name, 'val', 'noisyinput', filename+'.'+suffix)
        
        with open(noisyinput_file, 'rb') as f:
            plydata = PlyData.read(f)
            length = len(plydata.elements[0].data)
            data = plydata.elements[0].data
            data_pd = pd.DataFrame(data)
            noisyinput = np.zeros(data_pd.shape, dtype=np.float64)
            property_names = data[0].dtype.names

            for x, name in enumerate(property_names):
                noisyinput[:, x] = data_pd[name]

        max_len = noisyinput.shape[0]

        ### concat patches to one numpy file     
        with open(rec_file, 'w') as fw:
            file_path = os.path.join(blocks_path, filelist[start])
            with open(file_path, 'rb') as f:
                plydata = PlyData.read(f)
                length = len(plydata.elements[0].data)
                data = plydata.elements[0].data
                data_pd = pd.DataFrame(data)
                combine_pc = np.zeros(data_pd.shape, dtype=np.float64)
                property_names = data[0].dtype.names

                for y, name in enumerate(property_names):
                    combine_pc[:, y] = data_pd[name]
            
            for j in range(start+1, end+1):
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
                combine_pc = np.concatenate((combine_pc, data_np))
                
            fused_pc = create_dic_fuse(noisyinput, combine_pc)                                         
            # write header for point cloud
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
            fw.write('property float red\n')
            fw.write('property float green\n')
            fw.write('property float blue\n')
            fw.write('end_header\n')   

            for h in range(len(fused_pc)):
                    x = fused_pc[h][0]
                    y = fused_pc[h][1]
                    z = fused_pc[h][2]
                    r = fused_pc[h][3]
                    g = fused_pc[h][4]
                    b = fused_pc[h][5]
                    line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
                    fw.write(line + '\n')
            if(end + 1 != len(filelist)):
                # update index, current frame and total lines 
                start = i
                current_frame = filelist[start].split('_')[-3]
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
