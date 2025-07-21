import open3d as o3d
import time, os, sys, glob, argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import torch
import MinkowskiEngine as ME
import multiprocessing as mp
from model.Network import MyNet
from utils.myloss import myLoss

from data import make_data_loader
from utils.loss import get_metrics,k_nearest_neighbors
from utils.pc_error_wrapper import pc_error

from tensorboardX import SummaryWriter
from plyfile import *
import logging


def getlogger(logdir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default='/home/jupyter-austin2/pc_dataset/soldier_block_val/')
    parser.add_argument("--downsample", default=1, help='Downsample Rate')
    parser.add_argument("--num_test", type=int, default=1024, help='how many of the dataset use for testing')
    
    parser.add_argument("--dataset_8i", default='/home/jupyter-austin2/pc_dataset/MIR_Dataset/ply_block42_4096/val/noisyinput/')
    parser.add_argument("--dataset_8i_GT", default='/home/jupyter-austin2/pc_dataset/MIR_Dataset/ply_block42_4096/val/ground/')
    parser.add_argument("--dataset_8i_val", default='/home/jupyter-austin2/pc_dataset/MIR_Dataset/ply_block42_4096/train/noisyinput/')
    parser.add_argument("--dataset_8i_val_GT", default='/home/jupyter-austin2/pc_dataset/MIR_Dataset/ply_block42_4096/train/ground/')
    
    
    # for testing
#     parser.add_argument("--dataset_8i", default='/home/jupyter-austin2/pc_dataset/xyz(fixed)_rgb(noise)/test_train/')
#     parser.add_argument("--dataset_8i_GT", default='/home/jupyter-austin2/pc_dataset/xyz(fixed)_rgb(noise)/test_train_gt/')
#     parser.add_argument("--dataset_8i_val", default='/home/jupyter-austin2/pc_dataset/xyz(fixed)_rgb(noise)/test_val/')
#     parser.add_argument("--dataset_8i_val_GT", default='/home/jupyter-austin2/pc_dataset/xyz(fixed)_rgb(noise)/test_val_gt/')
    parser.add_argument("--last_kernel_size", type=int, default=5,
                        help='The final layer kernel size, coordinates get expanded by this')

    parser.add_argument("--init_ckpt", default='')  # ckpts/8x_0x_ks7/iter2000.pth
    parser.add_argument("--reset", default=False, action='store_true', help='reset training')

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--global_step", type=int, default=int(40000))
    parser.add_argument("--base_step", type=int, default=int(100), help='frequency for recording state.')
    parser.add_argument("--test_step", type=int, default=int(1), help='frequency for test and save.')
    # parser.add_argument("--random_seed", type=int, default=4, help='random_seed.')

    parser.add_argument("--max_norm", type=float, default=1, help='max norm for gradient clip, close if 0')
    parser.add_argument("--clip_value", type=float, default=0, help='max value for gradient clip, close if 0')

    parser.add_argument("--logdir", type=str, default='logs', help="logger direction.")
    parser.add_argument("--ckptdir", type=str, default='/home/jupyter-austin2/denoising_model', help="ckpts direction.")
    parser.add_argument("--prefix", type=str, default='test_loss', help="prefix of checkpoints/logger, etc.")
    parser.add_argument("--save_loc", default='/home/jupyter-austin2/Result/')
    parser.add_argument("--rec_loc", default='/home/jupyter-austin2/pc_dataset/combine_ply/output/')
    parser.add_argument("--test_loss_loc", default='/home/jupyter-austin2/val_loss/')
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="gamma for lr_scheduler.")
    parser.add_argument("--lr_step", type=int, default=20000, help="step for adjusting lr_scheduler.")

    args = parser.parse_args()
    return args


def kdtree_partition2(pc, pc2, max_num):
    parts = []
    parts2 = []

    class KD_node:
        def __init__(self, point=None, LL=None, RR=None):
            self.point = point
            self.left = LL
            self.right = RR

    def createKDTree(root, data, data2):
        if len(data) <= max_num:
            parts.append(data)
            parts2.append(data2)
            return

        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]
        data2_sorted = data2[np.lexsort(data2.T[dim_index, None])]

        point = data_sorted[int(len(data) / 2)]

        root = KD_node(point)
        root.left = createKDTree(root.left, data_sorted[: int((len(data) / 2))], data2_sorted[: int((len(data2) / 2))])
        root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):], data2_sorted[int((len(data2) / 2)):])

        return root

    init_root = KD_node(None)
    _ = createKDTree(init_root, pc, pc2)

    return parts, parts2


### This testing function is made for larger point clouds and uses kd_tree partition.
def test2(model, test_dataloader, logger, writer, writername, step, test_pc_error, args, device, First_eval):
    start_time = time.time()

    # data.
    test_iter = iter(test_dataloader)

    # loss & metrics.
    sum_loss = 0.
    all_metrics = np.zeros((1, 3))
    all_pc_errors = np.zeros(3)
    all_pc_errors2 = np.zeros(2)

    # model & crit.
    model.to(device)  # to cpu.
    # criterion.
    crit = torch.nn.BCEWithLogitsLoss()

    # loop per batch.
    for i in range(len(test_iter)):
        coords, _, coords_T = test_iter.next()
        parts_pc, parts_pc2 = kdtree_partition2(coords[:, 1:].numpy(), coords_T[:, 1:].numpy(), 70000)
        out_l = []
        out_cls_l = []
        target_l = []
        keep_l = []

        # Forward.
        for j, pc in enumerate(parts_pc):
            p = ME.utils.batched_coordinates([pc])
            p2 = ME.utils.batched_coordinates([parts_pc2[j]])
            f = torch.from_numpy(np.vstack(np.expand_dims(np.ones(p.shape[0]), 1))).float()
            x1 = ME.SparseTensor(features=f.cuda(), coordinates=p.cuda())

            with torch.no_grad():
                out, out_cls, target= model(x1, coords_T=p2, device=device, prune=True)

            out_l.append(out.C[:, 1:])
            out_cls_l.append(out_cls.F)
            target_l.append(target)
#             keep_l.append(keep)

        rec_pc = torch.cat(out_l, 0)
        rec_pc_cls = torch.cat(out_cls_l, 0)
        rec_target = torch.cat(target_l, 0)
#         rec_keep = torch.cat(keep_l, 0)

        loss = crit(rec_pc_cls.squeeze(), rec_target.type(out_cls.F.dtype).to(device))
#         metrics = get_metrics(rec_keep, rec_target)

        # get pc_error.
        if test_pc_error:
            GT_pcd = o3d.geometry.PointCloud()
            GT_pcd.points = o3d.utility.Vector3dVector(coords_T[:, 1:])
            # ori_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=20))

            GTfile = 'tmp/' + args.prefix + 'GT.ply'
            o3d.io.write_point_cloud(GTfile, GT_pcd, write_ascii=True)

            rec_pcd = o3d.geometry.PointCloud()
            rec_pcd.points = o3d.utility.Vector3dVector(rec_pc.cpu().numpy())

            recfile = 'tmp/' + args.prefix + 'rec.ply'
            o3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)

            pc_error_metrics = pc_error(infile1=GTfile, infile2=recfile, res=1024)
            pc_errors = [pc_error_metrics['mse1,PSNR (p2point)'][0],
                         pc_error_metrics['mse2,PSNR (p2point)'][0],
                         pc_error_metrics['mseF,PSNR (p2point)'][0]]

            if First_eval:
                in_pcd = o3d.geometry.PointCloud()
                in_pcd.points = o3d.utility.Vector3dVector(coords[:, 1:].numpy())

                infile = 'tmp/' + args.prefix + 'in.ply'
                o3d.io.write_point_cloud(infile, in_pcd, write_ascii=True)

                pc_error_metrics = pc_error(infile1=GTfile, infile2=infile, res=1024)
                pc_errors2 = [pc_error_metrics['mse1,PSNR (p2point)'][0],
                              pc_error_metrics['mseF,PSNR (p2point)'][0]]

        # record.
        with torch.no_grad():
            sum_loss += loss.item()
#             all_metrics += np.array(metrics)
            if test_pc_error:
                all_pc_errors += np.array(pc_errors)
                if First_eval:
                    all_pc_errors2 += np.array(pc_errors2)

    print('======testing time:', round(time.time() - start_time, 4), 's')

    sum_loss /= len(test_iter)
#     all_metrics /= len(test_iter)
    if test_pc_error:
        all_pc_errors /= len(test_iter)
        if First_eval:
            all_pc_errors2 /= len(test_iter)

    # logger.
    logger.info(f'\nIteration: {step}')
    logger.info(f'Sum Loss: {sum_loss:.4f}')
#     logger.info(f'Metrics (prec, recal, IOU): {np.round(all_metrics, 4).tolist()}')
    if test_pc_error:
        logger.info(f'all_pc_errors: {np.round(all_pc_errors, 4).tolist()}')

    # writer.
    writer.add_scalars(main_tag=writername + '/losses',
                       tag_scalar_dict={'sum_loss': sum_loss},
                       global_step=step)

#     writer.add_scalars(main_tag=writername + '/metrics',
#                        tag_scalar_dict={'Precision': all_metrics[0, 0],
#                                         'Recall': all_metrics[0, 1],
#                                         'IoU': all_metrics[0, 2]},
#                        global_step=step)

    if test_pc_error:
        writer.add_scalars(main_tag=writername + '/out_pc_errors',
                           tag_scalar_dict={'p2point1': all_pc_errors[0],
                                            'p2point2': all_pc_errors[1],
                                            'p2pointF': all_pc_errors[2], },
                           global_step=step)
        if First_eval:
            writer.add_scalars(main_tag=writername + '/In_pc_errors',
                               tag_scalar_dict={'p2point1': all_pc_errors2[0],
                                                'p2pointF': all_pc_errors2[1]},
                               global_step=step)
    return


### This testing function is made for smaller point clouds and does not use kd_tree partition.
def test1(model, test_dataloader, logger, writer, writername, step, args, device):
    start_time = time.time()

    test_loss_dir = os.path.join(args.test_loss_loc, args.prefix)
#     if not os.path.exists(test_loss_dir):
#         os.makedirs(test_loss_dir)
        
    # data.
    test_iter = iter(test_dataloader)

    # loss & metrics.
    sum_loss = 0
#     sum_loss_R = 0
#     sum_loss_G = 0
#     sum_loss_B = 0
#     all_metrics = np.zeros((1, 3))

    # model & crit.
    model.to(device)  # to cpu.
    # criterion.
#     crit = torch.nn.BCEWithLogitsLoss()
    crit = torch.nn.MSELoss()    

    # loop per batch.
    for i in range(len(test_iter)):
        try:
            coords, feats, coords_T, filedir = test_iter.next()
        except StopIteration:
            test_iter = iter(test_dataloader)
            coords, feats, coords_T, filedir = test_iter.next()

            
### coords normalize        
        ### coords normalize
        coords_copy, coordsT_copy = feats.float().numpy(), coords_T.float().numpy()

        #normalize
        scaler, coords_norm = min_max_normalize(coords_copy, 64)
        scalerT, coordsT_norm = min_max_normalize(coordsT_copy, 64)

        coords_norm_tensor = torch.from_numpy(coords_norm)
        coordsT_norm_tensor = torch.from_numpy(coordsT_norm)
### yuv

#         y = 0.2126*feats[:, 0] + 0.7152*feats[:, 1] + 0.0722*feats[:, 2]
#         y_re = y.reshape(-1, 1)

#         u = -0.1146*feats[:, 0] - 0.3854*feats[:, 1] + 0.5000*feats[:, 2]
#         u_re = u.reshape(-1, 1)

#         v = 0.5000*feats[:, 0] - 0.4542*feats[:, 1] - 0.0458*feats[:, 2]
#         v_re = v.reshape(-1, 1)
        
#         y_gt = 0.2126*coords_T[:, 0] + 0.7152*coords_T[:, 1] + 0.0722*coords_T[:, 2]
#         y_gt_re = y_gt.reshape(-1, 1)

#         ### yuv
#         u_tensor = ME.SparseTensor(features=u_re.cuda(), coordinates=coords.cuda())
#         v_tensor = ME.SparseTensor(features=v_re.cuda(), coordinates=coords.cuda())        

        
        
### yuv
#         u_tensor = ME.SparseTensor(features=u.cuda(), coordinates=coords.cuda())
#         v_tensor = ME.SparseTensor(features=v.cuda(), coordinates=coords.cuda())
        
#         coords_gt = ME.SparseTensor(features=y_gt_re.cuda(), coordinates=coords, device=device, requires_grad=True)
        x = ME.SparseTensor(features=coords_norm_tensor.cuda(), coordinates=coords.cuda())
        sp_coordsT_norm = coordsT_norm_tensor.cuda()
        loss = crit(x.F, sp_coordsT_norm.type(torch.float).to(device))
	
        # Forward.
        with torch.no_grad():
            out, out_cls = model(x, coords_T=coords_T, device=device, prune=True)

#         loss = crit(out_cls.F.squeeze(), target.type(out_cls.F.dtype).to(device))
        
#         loss_r = crit(x.F[:, 0], coords_gt.F[:, 0].type(torch.float).to(device))
#         loss_g = crit(x.F[:, 1], coords_gt.F[:, 1].type(torch.float).to(device))
#         loss_b = crit(x.F[:, 2], coords_gt.F[:, 2].type(torch.float).to(device))
#         metrics = get_metrics(keep, target)

        # record.
        with torch.no_grad():
            sum_loss += loss.item()
#             sum_loss_R += loss_r.item()
#             sum_loss_G += loss_g.item()
#             sum_loss_B += loss_b.item()
            
#             all_metrics += np.array(metrics)

####	restruct ply file                
#         fixed_coords = x.C.cpu().numpy()
#         out_coords = out_cls.C.cpu().numpy()
        
#         ### yuv
#         out_y = out_cls.F.cpu().detach().numpy()
#         out_u = u_tensor.F.cpu().numpy()
#         out_v = v_tensor.F.cpu().numpy()

# #         # yuv --> rgb   
# # #         out_y = out_cls.F
# # #         out_r = out_y + 1.402*v_tensor.F
# # #         out_g = out_y - 0.344*u_tensor.F - 0.792*v_tensor.F
# # #         out_b = out_y + 1.772*u_tensor.F
        
#         out_path = os.path.join(args.save_loc, args.prefix, str(step))
#         if not os.path.exists(out_path):
#             os.makedirs(out_path)	
#         filedir_split = filedir[0].split('/')
#         recfile = os.path.join(out_path, filedir_split[-1])
# #         print(recfile)
	
#         with open(recfile, 'w') as f:
#             f.write('ply\n')
#             f.write('format ascii 1.0\n')
#             f.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
#             f.write('comment frame_to_world_scale 0.181731\n')
#             f.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
#             f.write('comment width 1023\n')
#             f.write('element vertex ' + str(len(out_coords)) + '\n')
#             f.write('property float x\n')
#             f.write('property float y\n')
#             f.write('property float z\n')
#             f.write('property float red\n')
#             f.write('property float green\n')
#             f.write('property float blue\n')
#             f.write('end_header\n')
#             for i in range(len(out_coords)):
#                 x = fixed_coords[i][1]
#                 y = fixed_coords[i][2]
#                 z = fixed_coords[i][3]
#                 r = out_y[i][0]
#                 g = out_u[i][0]
#                 b = out_v[i][0]
#                 line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
#                 f.write(line + '\n')  
                
                
#     print("==================block combining=================")
#     rec_path = os.path.join(args.rec_loc, args.prefix, str(step))

#     if not os.path.exists(rec_path):
#         os.makedirs(rec_path)

#     filelist = os.listdir(out_path)
#     filelist.sort(key=lambda x : int(x.split('.')[0].split('_')[-2]))

#     rec_pc = []

#     current_frame = filelist[0].split('_')[-2]
#     start = 0
#     end = 0
#     tot_length = 0
#     for i in range (len(filelist) + 1):
#         # check if we meet next frame or reach last frame, yes --> reconstruct the current frame
#         if(i == len(filelist) or (filelist[i].split('_')[-2] != current_frame)):
#             print('Reconstruct Frame(points): ' + str(filelist[i-1].split('_')[-2]) + '(' + str(tot_length) + ')')
#             end = i - 1
#             suffix = filelist[end].split('.')[-1]
#             filename = "_".join(filelist[end].split('_')[:-1])
#             rec_file = rec_path + '/' + filename + '.' + suffix
#             with open(rec_file, 'w') as fw:
#                 for j in range(start, end + 1):
#                     # write header for point cloud
#                     if(j == start):
#                         fw.write('ply\n')
#                         fw.write('format ascii 1.0\n')
#                         fw.write('comment Version 2, Copyright 2017, 8i Labs, Inc.\n')
#                         fw.write('comment frame_to_world_scale 0.181731\n')
#                         fw.write('comment frame_to_world_translation -39.1599 3.75652 -46.6228\n')
#                         fw.write('comment width 1023\n')
#                         fw.write('element vertex ' + str(tot_length) +'\n')
#                         fw.write('property float x\n')
#                         fw.write('property float y\n')
#                         fw.write('property float z\n')
#                         fw.write('property float red\n')
#                         fw.write('property float green\n')
#                         fw.write('property float blue\n')
#                         fw.write('end_header\n')

#                     #write data for point cloud    
#                     file_path = os.path.join(out_path, filelist[j])
#                     with open(file_path, 'rb') as f:
#                         plydata = PlyData.read(f)
#                         length = len(plydata.elements[0].data)
#                         data = plydata.elements[0].data
#                         data_pd = pd.DataFrame(data)
#                         data_np = np.zeros(data_pd.shape, dtype=np.float64)
#                         property_names = data[0].dtype.names

#                         for y, name in enumerate(property_names):
#                             data_np[:, y] = data_pd[name]


#                     for h in range(len(data_np)):
#                             x = data_np[h][0]
#                             y = data_np[h][1]
#                             z = data_np[h][2]
#                             r = data_np[h][3]
#                             g = data_np[h][4]
#                             b = data_np[h][5]
#                             line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r) + ' ' + str(g) + ' ' + str(b)
#                             fw.write(line + '\n')


#             if(end + 1 != len(filelist)):
#                 # update index, current frame and total lines 
#                 start = i
#                 current_frame = filelist[start].split('_')[-2]
#                 tot_length = 0

#         if(i != len(filelist)):
#             file_path = os.path.join(out_path, filelist[i])
#             with open(file_path, 'rb') as f1:
#                 plydata = PlyData.read(f1)
#                 length = len(plydata.elements[0].data)
#                 data = plydata.elements[0].data
#                 data_pd = pd.DataFrame(data)

#                 # sum up total lines for each point cloud
#                 tot_length += data_pd.shape[0]


#     print('finished.')
	
    test_time = round(time.time() - start_time, 4)
    logger.info(f'======testing time:{test_time:.4f}s')

    sum_loss /= len(test_iter)
#     sum_loss_R /= len(test_iter)
#     sum_loss_G /= len(test_iter)
#     sum_loss_B /= len(test_iter)
#     all_metrics /= len(test_iter)

    # logger.
    logger.info(f'\nIteration: {step}')
    logger.info(f'Sum Loss: {sum_loss:.4f}')
#     logger.info(f'Red Channel Loss: {sum_loss_R:.4f}')
#     logger.info(f'Green Channel Loss: {sum_loss_G:.4f}')
#     logger.info(f'Blue Channel Loss: {sum_loss_B:.4f}')
#     logger.info(f'Metrics (prec, recal, IOU): {np.round(all_metrics, 4).tolist()}')

#### write test loss to .txt
    with open(test_loss_dir + ".txt", 'a') as fl:
        line = str(sum_loss)
        fl.write(line + '\n')

    # writer.
    writer.add_scalars(main_tag=writername + '/losses',
                       tag_scalar_dict={'sum_loss': sum_loss},
                       global_step=step)

#     writer.add_scalars(main_tag=writername + '/metrics',
#                        tag_scalar_dict={'Precision': all_metrics[0, 0],
#                                         'Recall': all_metrics[0, 1],
#                                         'IoU': all_metrics[0, 2]},
#                        global_step=step)
    
    return


def train(model, train_dataloader, test_dataloader2, logger, writer, args, device):
    # Optimizer.
    optimizer = torch.optim.Adam([{"params": model.parameters(), 'lr': args.lr}],
                                 betas=(0.9, 0.999), weight_decay=1e-4)
    # adjust lr.
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    # criterion.
#     crit = torch.nn.BCEWithLogitsLoss()
    crit = torch.nn.MSELoss()

    # define checkpoints direction.
    ckptdir = os.path.join(args.ckptdir, args.prefix)
    if not os.path.exists(ckptdir):
        logger.info(f'Make direction for saving checkpoints: {ckptdir}')
        os.makedirs(ckptdir)

    # Load checkpoints.
    start_step = 1
    if args.init_ckpt == '':
        logger.info('Random initialization.')
        First_eval = True
    else:
        # load params from checkpoints.
        logger.info(f'Load checkpoint from {args.init_ckpt}')
        ckpt = torch.load(args.init_ckpt)
        model.load_state_dict(ckpt['model'])
        First_eval = False
        # load start step & optimizer.
        if not args.reset:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except:
                print("Optimizer State Load Failed")
            start_step = ckpt['step'] + 1

    # start step.
    logger.info(f'LR: {scheduler.get_lr()}')
    print('==============', start_step)

    train_iter = iter(train_dataloader)
    start_time = time.time()
    sum_loss = 0
#     sum_loss_R = 0
#     sum_loss_G = 0
#     sum_loss_B = 0
    all_metrics = np.zeros((1, 3))

    for i in range(start_step, args.global_step + 1):
        if i % 10 == 0:
            print(i)
        optimizer.zero_grad()

        s = time.time()
#         for i in range(len(train_iter)):
        try:
            coords, feats, coords_T, filedir = train_iter.next()
        except StopIteration:
            train_iter = iter(train_dataloader)
            coords, feats, coords_T, filedir = train_iter.next()
        dataloader_time = time.time() - s

#         single_feat = feats[:, 2].reshape(-1,1)
#         single_featgt = coords_T[:, 2].reshape(-1, 1)
### coords normalize        
        ### coords normalize
        coords_copy, coordsT_copy = feats.float().numpy(), coords_T.float().numpy()

        #normalize
        scaler, coords_norm = min_max_normalize(coords_copy, 64)
        scalerT, coordsT_norm = min_max_normalize(coordsT_copy, 64)

        coords_norm_tensor = torch.from_numpy(coords_norm)
        coordsT_norm_tensor = torch.from_numpy(coordsT_norm)
### yuv
#         y = 0.2126*feats[:, 0] + 0.7152*feats[:, 1] + 0.0722*feats[:, 2]
#         y_re = y.reshape(-1, 1)

#         y_gt = 0.2126*coords_T[:, 0] + 0.7152*coords_T[:, 1] + 0.0722*coords_T[:, 2]
#         y_gt_re = y_gt.reshape(-1, 1)

        x = ME.SparseTensor(features=coords_norm_tensor.cuda(), coordinates=coords.cuda())
        sp_coordsT_norm = coordsT_norm_tensor.cuda()

        if x.__len__() >= 1e10:
            logger.info(f'\n\n\n======= larger than 1e10 ======: {x.__len__()}\n\n\n')
            continue
        # Forward.
        _, out_cls = model(x, coords_T=coords_T, device=device, prune=False)
    
#         loss = crit(out_cls.F.squeeze(), target.type(out_cls.F.dtype).to(device))
        loss = crit(out_cls.F, sp_coordsT_norm.type(torch.float).to(device))
#         loss_r = crit(out_cls.F[:, 0], coords_gt.F[:, 0].type(torch.float).to(device))
#         loss_g = crit(out_cls.F[:, 1], coords_gt.F[:, 1].type(torch.float).to(device))
#         loss_b = crit(out_cls.F[:, 2], coords_gt.F[:, 2].type(torch.float).to(device))
#         metrics = get_metrics(keep, target)
        if torch.isnan(loss) or torch.isinf(loss):
            logger.info(f'\n== loss is nan ==, Step: {i}\n')
            continue

        # Backward.
#         loss.retain_grad()
#         loss.requires_grad_(True)
        loss.backward()

        # Optional clip gradient.
        if args.max_norm != 0:
            # clip by norm
            max_grad_before = max(p.grad.data.abs().max() for p in model.parameters())
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

            if total_norm > args.max_norm:

                def get_total_norm(parameters, norm_type=2):
                    total_norm = 0.
                    for p in parameters:
                        param_norm = p.grad.data.norm(norm_type)
                        total_norm += param_norm.item() ** norm_type
                    total_norm = total_norm ** (1. / norm_type)
                    return total_norm

                print('total_norm:',
                      '\nBefore: total_norm:,', total_norm,
                      'max grad:', max_grad_before,
                      '\nthreshold:', args.max_norm,
                      '\nAfter:', get_total_norm(model.parameters()),
                      'max grad:', max(p.grad.data.abs().max() for p in model.parameters()))

        if args.clip_value != 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_value)
            print('after gradient clip', max(p.grad.data.abs().max() for p in model.parameters()))

        # record.
        with torch.no_grad():
            sum_loss += loss.item()
#             sum_loss_R += loss_r.item()
#             sum_loss_G += loss_g.item()
#             sum_loss_B += loss_b.item()
            
#             all_metrics += np.array(metrics)

        optimizer.step()

        # Display.
        if i % args.base_step == 0:
            # average.
            with torch.no_grad():
                sum_loss /= args.base_step
#                 sum_loss_R /= args.base_step
#                 sum_loss_G /= args.base_step
#                 sum_loss_B /= args.base_step
#                 all_metrics /= args.base_step

            if np.isinf(sum_loss):
                logger.info('inf error!')
                sys.exit(0)

            if np.isnan(sum_loss):
                logger.info('NaN error!')
                sys.exit(0)

            # logger.
            logger.info(f'\nIteration: {i}')
            logger.info(f'Running time: {((time.time() - start_time) / 60):.2f} min')
            logger.info(f'Data Loading time: {dataloader_time:.5f} s')
            logger.info(f'LR: {scheduler.get_lr()}')
            logger.info(f'Sum Loss: {sum_loss:.4f}')
#             logger.info(f'Red Channel Loss: {sum_loss_R:.4f}')
#             logger.info(f'Green Channel Loss: {sum_loss_G:.4f}')
#             logger.info(f'Blue Channel Loss: {sum_loss_B:.4f}')
            
#             logger.info(f'Metrics (prec, recal, IOU): {np.round(all_metrics, 4).tolist()}')

            # writer.
            writer.add_scalars(main_tag='train/losses',
                               tag_scalar_dict={'sum_loss': sum_loss},
                               global_step=i)

#             writer.add_scalars(main_tag='train/metrics',
#                                tag_scalar_dict={'Precision': all_metrics[0, 0],
#                                                 'Recall': all_metrics[0, 1],
#                                                 'IoU': all_metrics[0, 2]},
#                                global_step=i)

            # return 0.
            sum_loss = 0
#             sum_loss_R = 0
#             sum_loss_G = 0
#             sum_loss_B = 0
            
#             all_metrics = np.zeros((1, 3))

            # empty cache.
            torch.cuda.empty_cache()

        if i % (args.test_step) == 0:
           # Evaluation 8i.
            logger.info(f'\n=====Evaluation: iter {i} 8i =====')
            
            # save.
            logger.info(f'save checkpoints: {ckptdir}/iter{str(i)}')
            torch.save({'step': i, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        }, os.path.join(ckptdir, 'iter' + str(i) + '.pth'))

            
            with torch.no_grad():
                 test1(model=model, test_dataloader=test_dataloader2,
                      logger=logger, writer=writer, writername='eval', step=i, args=args, device=device)
            torch.cuda.empty_cache()
            break

            model.to(device)

        scheduler.step()


    writer.close()


if __name__ == '__main__':
    args = parse_args()

    logdir = os.path.join(args.logdir, args.prefix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    logger = getlogger(logdir)
    logger.info(args)
    writer = SummaryWriter(log_dir=logdir)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device:{device}')

#     # Load data.
#     filedirs = glob.glob(args.dataset + '*.h5')
#     filedirs = sorted(filedirs)
#     logger.info(f'Files length: {len(filedirs)}')
    
    # 8i dataset training
    eighti_filedirs = glob.glob(args.dataset_8i + '*.ply')
    eighti_filedirs = sorted(eighti_filedirs)
    logger.info(f'Trainin Files length: {len(eighti_filedirs)}')
    
    # 8i dataset training GT
    eighti_GT_filedirs = glob.glob(args.dataset_8i_GT + '*.ply')
    eighti_GT_filedirs = sorted(eighti_filedirs)
    logger.info(f'GT Trainin Files length: {len(eighti_GT_filedirs)}')

    # 8i dataset validation
    eighti_filedirs_val = glob.glob(args.dataset_8i_val + '*.ply')
    eighti_filedirs_val = sorted(eighti_filedirs_val)
    logger.info(f'Validation Files length: {len(eighti_filedirs_val)}')
    
    # 8i dataset validation GT
    eighti_GT_filedirs_val = glob.glob(args.dataset_8i_val_GT + '*.ply')
    eighti_GT_filedirs_val = sorted(eighti_filedirs_val)
    logger.info(f'GT Validation Files length: {len(eighti_GT_filedirs_val)}')
    
#     train_dataloader = make_data_loader(files=filedirs[int(args.num_test):],
#                                         GT_folder=None,
#                                         batch_size=args.batch_size,
#                                         downsample=args.downsample,
#                                         shuffle=True,
#                                         num_workers=mp.cpu_count(),
#                                         repeat=True)

    train_dataloader = make_data_loader(files=eighti_filedirs,
                                        GT_folder=args.dataset_8i_GT,
                                        batch_size=args.batch_size,
                                        downsample=args.downsample,
                                        shuffle=True,
                                        num_workers=mp.cpu_count(),
                                        repeat=False)

#     test_dataloader = make_data_loader(files=filedirs[:int(args.num_test)],
#                                        GT_folder=None,
#                                        batch_size=args.batch_size,
#                                        downsample=args.downsample,
#                                        shuffle=False,
#                                        num_workers=mp.cpu_count(),
#                                        repeat=False)

    # 8i dataset
    # eighti_filedirs = glob.glob(args.dataset_8i + '*.ply')
    # eighti_filedirs = sorted(eighti_filedirs)
    # logger.info(f'Files length: {len(eighti_filedirs)}')

    eighti_dataloader = make_data_loader(files=eighti_filedirs_val,
                                        GT_folder=args.dataset_8i_val_GT  ,
                                        batch_size=1,
                                        downsample=args.downsample,
                                        shuffle=False,
                                        num_workers=1,
                                        repeat=False)

#     test_dataloader2 = make_data_loader(files=filedirs[int(args.num_test):],
#                                          GT_folder=None,
#                                          batch_size=args.batch_size,
#                                          downsample=args.downsample,
#                                          shuffle=False,
#                                          num_workers=mp.cpu_count(),
#                                          repeat=False)

    # Network.
    model = MyNet(last_kernel_size=args.last_kernel_size).to(device)
#     logger.info(model)
#     model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
#     model = model.cuda(device=device_ids[0])
    
    train(model, train_dataloader, eighti_dataloader, logger, writer, args, device)
