import open3d as o3d
import os, glob, argparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pandas as pd
import torch
import MinkowskiEngine as ME
from plyfile import *
from model.Network_2 import MyNet
from data import make_data_loader
from utils.norm import normalize, unnormalize, min_max_normalize, min_max_unnormalize
# from utils.utility import get_loss, my_knn_query
from knn_cuda import KNN

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # For 8x upsampling
    parser.add_argument("--pretrained", default='/home/jupyter-austin2/pc_dataset/denoising_model/MEtest8i_qp48_interpolate30_cube70_lr0001_batchsize4/iter5000.pth', help='Path to pretrained model')
    # For 4x upsampling
    # parser.add_argument("--pretrained", default='./ckpts/4x_0x_ks5/iter64000.pth', help='Path to pretrained model')
    parser.add_argument("--prefix", type=str, default='MEtest8i_qp48_interpolate30_cube70_lr0001_batchsize4', help="prefix of checkpoints/logger, etc.")
    parser.add_argument("--up_ratio", default=1, help='Upsample Ratio')

    parser.add_argument("--test_dataset", default='/home/jupyter-austin2/pc_dataset/Complement_dataset/ply_interpolation/ply_qp48_cube70/interpolate_30/val/noisyinput/')
    parser.add_argument("--save_loc", default='/home/jupyter-austin2/pc_dataset/Complement_dataset/coordenhanced_ply_interpolation/ply_qp48_cube70/interpolate_30/val/noisyinput/')
    parser.add_argument("--rec_loc", default='/home/jupyter-austin2/zx/result_rec/')
    parser.add_argument("--iter", default=135, help='which iteration reconstructs for')
    parser.add_argument("--last_kernel_size", type=int, default=7, help='The final layer kernel size, coordinates get expanded by this')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--padding_cube", type=int, default=400, help="padding size for smaller cube")
    parser.add_argument("--normalize", type=int, default=100, help="range for normalization")
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

# for model in os.listdir(args.pretrained):
# model_path = os.path.join(args.pretrained, model)
model_path = args.pretrained
# iters = model.split('.')[0]
iters = "1"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyNet(last_kernel_size=args.last_kernel_size).to(device)
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['model'])

l = len(test_files)

eighti_test_dataloader = make_data_loader(files=test_files,
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

    coords, coord_feats, feats, coords_T, filedir = test_iter.next()
#     print('==========feats size before putting in sparse tensor===========')
#     print(feats.shape)
    if args.batch_size == 1:
#         len_coord = coord_feats.shape[0]
#         num_coord_feats = coord_feats.cpu().detach().numpy()
#         num_coords = coords.cpu().detach().numpy()

#         if len_coord <= args.padding_cube:
#             pad_coord_feats = np.pad(num_coord_feats, ((0, (args.padding_cube-len_coord)), (0, 0)), mode="linear_ramp")
#             pad_coords = np.pad(num_coords, ((0, (args.padding_cube-len_coord)), (0, 0)), mode="linear_ramp")
#         else:
#             pad_coord_feats = num_coord_feats
#             pad_coords = num_coords

#         min_max_scaler, norm_coords = min_max_normalize(pad_coord_feats, args.normalize)
#         norm_coords_tensor = torch.from_numpy(norm_coords)
#         _, norm_coords_T = min_max_normalize(coords_T.cpu().detach().numpy(), args.normalize)
#         norm_coords_T_tensor = torch.from_numpy(norm_coords_T)

#         coords_tensor = torch.from_numpy(pad_coords)
        x = ME.SparseTensor(features=coord_feats, coordinates=coords, device=device)

        # network forward
        out, out_cls = model(x, coords_T=coords_T, device=device, prune=False)

#         generated_norm_coords = out_cls.F.cpu().detach().numpy()
#         if len_coord <= args.padding_cube:
#             generated_norm_coords = generated_norm_coords[:len_coord]
#         else:
#             generated_norm_coords = generated_norm_coords

#         generated_coords = min_max_unnormalize(min_max_scaler, generated_norm_coords)

#         generated_tensor = torch.from_numpy(generated_coords)
#             align_gen_knn = KNN(k=1, transpose_mode=True)
#             _, idx = align_gen_knn(coord_feats.float().unsqueeze(0).cuda(), generated_tensor.float().unsqueeze(0).cuda())
        color_align_KNN = KNN(k=1, transpose_mode=True)
        _, aligned_color_idx = color_align_KNN(coord_feats.to(device).unsqueeze(0), out_cls.F.unsqueeze(0)) 
        generated_coords = out_cls.F.cpu().detach().numpy()
        
        aligned_color = np.zeros(generated_coords.shape)
        for i in range(generated_coords.shape[0]):
            aligned_color[i, 0] = feats[aligned_color_idx[0, i, 0]][0]
            aligned_color[i, 1] = feats[aligned_color_idx[0, i, 0]][1]
            aligned_color[i, 2] = feats[aligned_color_idx[0, i, 0]][2]

    else:
        min_max_scaler, norm_coords = min_max_normalize(coord_feats.cpu().detach().numpy(), args.normalize)
        norm_coords_tensor = torch.from_numpy(norm_coords)
        _, norm_coords_T = min_max_normalize(coords_T.cpu().detach().numpy(), args.normalize)
        norm_coords_T_tensor = torch.from_numpy(norm_coords_T)
        x = ME.SparseTensor(features=norm_coords_tensor.cuda(), coordinates=coords.cuda())

        # network forward
        out, out_cls = model(x, coords_T=coords_T, device=device, prune=False)

        generated_norm_coords = out_cls.F.cpu().detach().numpy()
#         if len_coord <= args.padding_cube:
#             generated_norm_coords = generated_norm_coords[:len_coord]
#         else:
#             generated_norm_coords = generated_norm_coords

        generated_coords = min_max_unnormalize(min_max_scaler, generated_norm_coords)

        generated_tensor = torch.from_numpy(generated_coords)
        align_gen_knn = KNN(k=1, transpose_mode=True)
        _, idx = align_gen_knn(coord_feats.float().unsqueeze(0).cuda(), generated_tensor.float().unsqueeze(0).cuda())
        gen_feats_tensor = torch.zeros(generated_coords.shape)

        for k in range(generated_coords.shape[0]):
            gen_feats_tensor[k] = feats[idx[0][k][0]]
        feats_num = gen_feats_tensor.numpy()


    ### yuv

#     y = 0.2126*feats[:, 0] + 0.7152*feats[:, 1] + 0.0722*feats[:, 2]
#     y_re = y.reshape(-1, 1)

#     u = -0.1146*feats[:, 0] - 0.3854*feats[:, 1] + 0.5000*feats[:, 2]
#     u_re = u.reshape(-1, 1)

#     v = 0.5000*feats[:, 0] - 0.4542*feats[:, 1] - 0.0458*feats[:, 2]
#     v_re = v.reshape(-1, 1)

#     ### yuv
#     u_tensor = ME.SparseTensor(features=u_re.cuda(), coordinates=coords.cuda())
#     v_tensor = ME.SparseTensor(features=v_re.cuda(), coordinates=coords.cuda())


#     x = ME.SparseTensor(features=norm_coords_tensor.cuda(), coordinates=coords, device=device)

    #     print(x.F.shape)
#     fixed_coords = x.C.cpu().numpy()

#     out, out_cls = model(x, coords_T=coords_T, device=device, prune=False)


#     out_coords = out_cls.C.cpu().numpy()

    ### yuv
#     out_y = out_cls.F.cpu().detach().numpy()
#     out_u = u_tensor.F.cpu().numpy()
#     out_v = v_tensor.F.cpu().numpy()


    ### yuv --> rgb   
#     out_y = out_cls.F
#     out_r = out_y + 1.402*v_tensor.F
#     out_g = out_y - 0.344*u_tensor.F - 0.792*v_tensor.F
#     out_b = out_y + 1.772*u_tensor.F

#     out_path = os.path.join(args.save_loc, iters)
    out_path = args.save_loc
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    filedir_split = filedir[0].split('/')
    recfile = os.path.join(out_path, filedir_split[-1])

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
                f.write('element vertex ' + str(len(generated_coords)) + '\n')
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
                f.write('end_header\n')
                for h in range(len(generated_coords)):
                    x = generated_coords[h][0]
                    y = generated_coords[h][1]
                    z = generated_coords[h][2]
                    r = int(aligned_color[h][0])
                    g = int(aligned_color[h][1])
                    b = int(aligned_color[h][2])
                    line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
                    f.write(line + '\n')  


# print("==================block combining=================")
# rec_path = os.path.join(args.rec_loc, args.prefix, iters)

# if not os.path.exists(rec_path):
#     os.makedirs(rec_path)

# filelist = os.listdir(out_path)
# filelist.sort(key=lambda x : int(x.split('.')[0].split('_')[-2]))

# rec_pc = []

# current_frame = filelist[0].split('_')[-2]
# start = 0
# end = 0
# tot_length = 0
# for i in range (len(filelist) + 1):
#     # check if we meet next frame or reach last frame, yes --> reconstruct the current frame
#     if(i == len(filelist) or (filelist[i].split('_')[-2] != current_frame)):
#         print('Reconstruct Frame(points): ' + str(filelist[i-1].split('_')[-2]) + '(' + str(tot_length) + ')')
#         end = i - 1
#         suffix = filelist[end].split('.')[-1]
#         filename = "_".join(filelist[end].split('_')[:-1])
#         rec_file = rec_path + '/' + filename + '.' + suffix
#         print(rec_file)
#         with open(rec_file, 'w') as fw:
#             for j in range(start, end + 1):
#                 # write header for point cloud
#                 if(j == start):
#                     fw.write('ply\n')
#                     fw.write('format ascii 1.0\n')
#                     fw.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
#                     fw.write('comment frame_to_world_scale 0.181731\n')
#                     fw.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
#                     fw.write('comment width 1023\n')
#                     fw.write('element vertex ' + str(tot_length) +'\n')
#                     fw.write('property float x\n')
#                     fw.write('property float y\n')
#                     fw.write('property float z\n')
#                     fw.write('property uchar red\n')
#                     fw.write('property uchar green\n')
#                     fw.write('property uchar blue\n')
#                     fw.write('end_header\n')

#                 #write data for point cloud    
#                 file_path = os.path.join(out_path, filelist[j])
#                 with open(file_path, 'rb') as f:
#                     plydata = PlyData.read(f)
#                     length = len(plydata.elements[0].data)
#                     data = plydata.elements[0].data
#                     data_pd = pd.DataFrame(data)
#                     data_np = np.zeros(data_pd.shape, dtype=np.float64)
#                     property_names = data[0].dtype.names

#                     for y, name in enumerate(property_names):
#                         data_np[:, y] = data_pd[name]


#                 for h in range(len(data_np)):
#                         x = data_np[h][0]
#                         y = data_np[h][1]
#                         z = data_np[h][2]
#                         r = int(data_np[h][3])
#                         g = int(data_np[h][4])
#                         b = int(data_np[h][5])
#                         line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
#                         fw.write(line + '\n')


#         if(end + 1 != len(filelist)):
#             # update index, current frame and total lines 
#             start = i
#             current_frame = filelist[start].split('_')[-2]
#             tot_length = 0

#     if(i != len(filelist)):
#         file_path = os.path.join(out_path, filelist[i])
#         with open(file_path, 'rb') as f1:
#             plydata = PlyData.read(f1)
#             length = len(plydata.elements[0].data)
#             data = plydata.elements[0].data
#             data_pd = pd.DataFrame(data)

#             # sum up total lines for each point cloud
#             tot_length += data_pd.shape[0]            

print('finished.')