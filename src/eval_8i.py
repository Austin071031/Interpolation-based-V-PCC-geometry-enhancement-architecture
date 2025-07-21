import open3d as o3d
import os, glob, argparse
import numpy as np
import pandas as pd
import torch
import MinkowskiEngine as ME
from plyfile import *
from model.Network import MyNet
from data import make_test_data_loader


def parse_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # For 8x upsampling
  parser.add_argument("--pretrained", default='/home/jupyter-austin2/denoising_model/ME_OLsoldier_500_y_batchsize8_lr0.001_lrstep400_unfreezall/iter135.pth', help='Path to pretrained model')
  # For 4x upsampling
  # parser.add_argument("--pretrained", default='./ckpts/4x_0x_ks5/iter64000.pth', help='Path to pretrained model')
  parser.add_argument("--prefix", type=str, default='ME_OLtrainsoldier_500_y_batchsize8_lr0.001_lrstep400_unfreezall', help="prefix of checkpoints/logger, etc.")
  parser.add_argument("--up_ratio", default=1, help='Upsample Ratio')
  
  parser.add_argument("--test_dataset", default='/home/jupyter-austin2/pc_dataset/online_learning/soldier/ply_block/train/kdtree_noisy_optimize_xyzunique/')
  parser.add_argument("--save_loc", default='/home/jupyter-austin2/Result/')
  parser.add_argument("--rec_loc", default='/home/jupyter-austin2/pc_dataset/combine_ply/output/')
  parser.add_argument("--iter", default=135, help='which iteration reconstructs for')
  parser.add_argument("--last_kernel_size", type=int, default=7, help='The final layer kernel size, coordinates get expanded by this')
  parser.add_argument("--batch_size", type=int, default=1)  
  args = parser.parse_args()
  return args


# def kdtree_partition(pc, max_num):
#     parts = []
    
#     class KD_node:  
#         def __init__(self, point=None, LL = None, RR = None):  
#             self.point = point  
#             self.left = LL  
#             self.right = RR
            
#     def createKDTree(root, data):
#         if len(data) <= max_num:
#             parts.append(data)
#             return
        
#         variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
#         dim_index = variances.index(max(variances))
#         data_sorted = data[np.lexsort(data.T[dim_index, None])]
        
#         point = data_sorted[int(len(data)/2)]  
        
#         root = KD_node(point)  
#         root.left = createKDTree(root.left, data_sorted[: int((len(data) / 2))])  
#         root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):]) 
        
#         return root
    
#     init_root = KD_node(None)
#     _ = createKDTree(init_root, pc)  
#     return parts

args = parse_args()
test_files = glob.glob(args.test_dataset+'*.ply')
test_files = sorted(test_files)
print("Test Files length: " + str(len(test_files)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyNet(last_kernel_size=args.last_kernel_size).to(device)
ckpt = torch.load(args.pretrained)
model.load_state_dict(ckpt['model'])

l = len(test_files)

save_path = os.path.join(args.save_loc, args.prefix, str(args.iter))
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
eighti_test_dataloader = make_test_data_loader(files=test_files,
                                    GT_folder=None  ,
                                    batch_size=args.batch_size,
                                    downsample=1,
                                    shuffle=False,
                                    num_workers=1,
                                    repeat=False)
test_iter = iter(eighti_test_dataloader)



# for i, pc in enumerate(test_files):
print("==================inferencing=================")
for i in range(len(eighti_test_dataloader)):
    print("!!!!!!!!==========!!!!!!!!!!  FILE NUMBER: ", i+1, ' / ', l)
    # loss & metrics.

#     file_name = os.path.basename(pc)
#     pcd = o3d.io.read_point_cloud(pc)
#     coords = np.asarray(pcd.points)
#     parts = kdtree_partition(coords, 70000)
    
    

#     out_list = []
#     for j,part in enumerate(parts):
#         p = ME.utils.batched_coordinates([part])
#         part_T = np.random.randint(0, part.max(), size=(part.shape[0]*args.up_ratio, 3))
        
#         p2 = ME.utils.batched_coordinates([part_T])
#         f = torch.from_numpy(np.vstack(np.expand_dims(np.ones(p.shape[0]), 1))).float()
#         x = ME.SparseTensor(features=f, coordinates=p, device=device)
#         with torch.no_grad():
#             out, _, _ = model(x, coords_T=p2, device=device, prune=True)
            
#         print(out.features[:,:].shape)
#         out_list.append(out.C[:,1:])

    coords, feats, coords_T, filedir = test_iter.next()
#     print('==========feats size before putting in sparse tensor===========')
#     print(feats.shape)
    
    
    ### yuv

    y = 0.2126*feats[:, 0] + 0.7152*feats[:, 1] + 0.0722*feats[:, 2]
    y_re = y.reshape(-1, 1)

    u = -0.1146*feats[:, 0] - 0.3854*feats[:, 1] + 0.5000*feats[:, 2]
    u_re = u.reshape(-1, 1)

    v = 0.5000*feats[:, 0] - 0.4542*feats[:, 1] - 0.0458*feats[:, 2]
    v_re = v.reshape(-1, 1)

    ### yuv
    u_tensor = ME.SparseTensor(features=u_re.cuda(), coordinates=coords.cuda())
    v_tensor = ME.SparseTensor(features=v_re.cuda(), coordinates=coords.cuda())

    
    x = ME.SparseTensor(features=y_re, coordinates=coords, device=device)
    
    #     print(x.F.shape)
    fixed_coords = x.C.cpu().numpy()
    
    out, out_cls = model(x, coords_T=coords_T, device=device, prune=False)
    
    
    out_coords = out_cls.C.cpu().numpy()
    
    ### yuv
    out_y = out_cls.F.cpu().detach().numpy()
    out_u = u_tensor.F.cpu().numpy()
    out_v = v_tensor.F.cpu().numpy()
    
    
    ### yuv --> rgb   
#     out_y = out_cls.F
#     out_r = out_y + 1.402*v_tensor.F
#     out_g = out_y - 0.344*u_tensor.F - 0.792*v_tensor.F
#     out_b = out_y + 1.772*u_tensor.F

    
    filedir_split = filedir[0].split('/')
    recfile = os.path.join(save_path, filedir_split[-1])
    
#     rec_pc = torch.cat(out, 0)
#     print("Number of points in input point cloud : ", coords.shape[0])
#     print("Number of points in output rec point cloud : ", rec_pc.cpu().numpy().shape[0])
#     print(rec_pc)
#     print(len(rec_pc.cpu().numpy()))

#     rec_pcd = o3d.geometry.PointCloud()
#     rec_pcd.points = o3d.utility.Vector3dVector(rec_pc)
#     o3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)


    with open(recfile, 'w') as f:
                f.write('ply\n')
                f.write('format ascii 1.0\n')
                f.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
                f.write('comment frame_to_world_scale 0.181731\n')
                f.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
                f.write('comment width 1023\n')
                f.write('element vertex ' + str(len(out_coords)) + '\n')
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
                f.write('property float red\n')
                f.write('property float green\n')
                f.write('property float blue\n')
                f.write('end_header\n')
                for i in range(len(out_coords)):
                    x = fixed_coords[i][1]
                    y = fixed_coords[i][2]
                    z = fixed_coords[i][3]
                    r = out_y[i][0]
                    g = out_u[i][0]
                    b = out_v[i][0]
                    line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
                    f.write(line + '\n')
                    
                    
print("==================block combining=================")
rec_path = os.path.join(args.rec_loc, args.prefix, str(args.iter))

if not os.path.exists(rec_path):
    os.makedirs(rec_path)
    
filelist = os.listdir(save_path)
filelist.sort(key=lambda x : int(x.split('.')[0].split('_')[-3]))

rec_pc = []

current_frame = filelist[0].split('_')[-3]
start = 0
end = 0
tot_length = 0
for i in range (len(filelist) + 1):
    # check if we meet next frame or reach last frame, yes --> reconstruct the current frame
    if(i == len(filelist) or (filelist[i].split('_')[-3] != current_frame)):
        print('Reconstruct Frame(points): ' + str(filelist[i-1].split('_')[-3]) + '(' + str(tot_length) + ')')
        end = i - 1
        suffix = filelist[end].split('.')[-1]
        filename = "_".join(filelist[end].split('_')[:-1])
        rec_file = rec_path + '/' + filename + '.' + suffix
        with open(rec_file, 'w') as fw:
            for j in range(start, end + 1):
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
                    fw.write('property float red\n')
                    fw.write('property float green\n')
                    fw.write('property float blue\n')
                    fw.write('end_header\n')

                #write data for point cloud    
                file_path = os.path.join(save_path, filelist[j])
                with open(file_path, 'rb') as f:
                    plydata = PlyData.read(f)
                    length = len(plydata.elements[0].data)
                    data = plydata.elements[0].data
                    data_pd = pd.DataFrame(data)
                    data_np = np.zeros(data_pd.shape, dtype=np.float64)
                    property_names = data[0].dtype.names

                    for y, name in enumerate(property_names):
                        data_np[:, y] = data_pd[name]


                for h in range(len(data_np)):
                        x = data_np[h][0]
                        y = data_np[h][1]
                        z = data_np[h][2]
                        r = data_np[h][3]
                        g = data_np[h][4]
                        b = data_np[h][5]
                        line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
                        fw.write(line + '\n')


        if(end + 1 != len(filelist)):
            # update index, current frame and total lines 
            start = i
            current_frame = filelist[start].split('_')[-3]
            tot_length = 0

    if(i != len(filelist)):
        file_path = os.path.join(save_path, filelist[i])
        with open(file_path, 'rb') as f1:
            plydata = PlyData.read(f1)
            length = len(plydata.elements[0].data)
            data = plydata.elements[0].data
            data_pd = pd.DataFrame(data)

            # sum up total lines for each point cloud
            tot_length += data_pd.shape[0]

                    
print('finished.')