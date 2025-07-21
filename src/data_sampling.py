# Options
#scale=10*2  #Voxel size = 1/scale
val_reps=64 # Number of test views, 1 or more
batch_size=1

from re import X
import torch, numpy as np, torch.utils.data, multiprocessing as mp, time

dimension=3
full_scale=2**6 #Input field size
output_dim=4

print('Reading Data from the files')
start = time.time()
train = np.load('/home/jupyter-austin2/pc_dataset/ME_dataset/soldier/ply/rgb/pointsets_train.npz', allow_pickle=True)
val = np.load('/home/jupyter-austin2/pc_dataset/ME_dataset/soldier/ply/rgb/pointsets_val.npz', allow_pickle=True)
gt_train = train['groundtruth']
input_train = train['noisyinput']

gt_val = val['groundtruth']
input_val = val['noisyinput']
print('Time taken to Load data: ', time.time()-start)
len_train = len(gt_train)
len_val  = len(gt_val)

print('Training Point Clouds:', len_train)
print('Validation Point Clouds:', len_val)
print('Number of CPU workers: ', mp.cpu_count())

# a = gt_train[1][:,3:]
# b = input_train[1][:,3:]
a = gt_train[0][:, 3:]
b = input_train[0][:, 3:]
print('ground:',a)
print('noisy:',b)
#def is_whole(d):
#    """Whether or not d is a whole number."""
#    return isinstance(d, int) or (isinstance(d, float) and d.is_integer())
#
#all(map(is_whole, xyz[:,2]))


def trainMerge(tbl):
    locs=[]
    feats=[]
    groundlocs=[]
    groundfeats=[]
    for idx,i in enumerate(tbl):
        a = gt_train[i][:,:3]#:3——》3：
        b = input_train[i][:,:3]
        choose = np.random.choice(len(a), 1)
        offset=-a[choose]+full_scale/2
        a+=offset
        idxs_gt=(a.min(1)>=0)*(a.max(1)<full_scale)
#         count_true_gt = 0
#         i_gt = 0
#         while(count_true_gt < 1024):
#             if(idxs[i_gt] == True):
#                 count_true_gt += 1
#             i_gt += 1
#         for j in range(i_gt, len(idxs)):
#             if(idxs[j] == True):
#                 idxs[j] = False

        a = a[idxs_gt]
        d = gt_train[i][:,3:]#rgb of groundtruth
        d = d[idxs_gt]       
        b+=offset
        idxs_input=(b.min(1)>=0)*(b.max(1)<full_scale)               
#         count_true_noisy = 0
#         i_noisy = 0
#         while(count_true_noisy < 1024):
#             if(idxs[i_noisy] == True):
#                 count_true_noisy += 1
#             i_noisy += 1
#         for j in range(i_noisy, len(idxs)):
#             if(idxs[j] == True):
#                 idxs[j] = False
        b=b[idxs_input]
        c = input_train[i][:,3:]#rgb of noisyinput
        c=c[idxs_input]
        a=torch.from_numpy(a).int()
        b=torch.from_numpy(b).int()
#         a=np.ones((len(a),1),dtype=np.float32)
#         b=np.ones((len(b),1),dtype=np.float32)
        c=torch.from_numpy(c).float()
        d=torch.from_numpy(d).float()
        locs.append(torch.cat([b,torch.IntTensor(b.shape[0],1).fill_(idx)],1))
        feats.append(torch.cat([c,torch.FloatTensor(c.shape[0],1).fill_(idx)],1))
#         feats.append(torch.from_numpy(b))
        groundlocs.append(torch.cat([a,torch.IntTensor(a.shape[0],1).fill_(idx)],1))
        groundfeats.append(torch.cat([d,torch.FloatTensor(d.shape[0],1).fill_(idx)],1))
        # groundfeats.append(torch.from_numpy(a))
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    groundlocs=torch.cat(groundlocs,0)
    groundfeats=torch.cat(groundfeats,0)
    
    # groundfeats=torch.cat(groundfeats,0)
    return {'x': [locs,feats], 'y': [groundlocs,groundfeats], 'id': tbl}
#     return locs, feats, groundlocs, groundfeats
train_data_loader = torch.utils.data.DataLoader(
    list(range(len(gt_train))),batch_size=batch_size, collate_fn=trainMerge, num_workers=0,shuffle=True)



def valMerge(tbl):
    locs=[]
    feats=[]
    groundlocs=[]
    groundfeats=[]
    for idx,i in enumerate(tbl):
        a = gt_train[i][:,:3]#:3——》3：
        b = input_train[i][:,:3]
        choose = np.random.choice(len(a), 1)
        offset=-a[choose]+full_scale/2
        
        #ground truth
        a+=offset
        idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
#         count_true_gt = 0
#         i_gt = 0
#         while(count_true_gt < 1024):
#             if(idxs[i_gt] == True):
#                 count_true_gt += 1
#             i_gt += 1
#         for j in range(i_gt, len(idxs)):
#             if(idxs[j] == True):
#                 idxs[j] = False
        a=a[idxs]
        d = gt_train[i][:,3:]#rgb of groundtruth
        d=d[idxs]
        
        # noisy input
        b+=offset
        idxs=(b.min(1)>=0)*(b.max(1)<full_scale)
#         count_true_noisy = 0
#         i_noisy = 0
#         while(count_true_noisy < 1024):
#             if(idxs[i_noisy] == True):
#                 count_true_noisy += 1
#             i_noisy += 1
#         for j in range(i_noisy, len(idxs)):
#             if(idxs[j] == True):
#                 idxs[j] = False
        b=b[idxs]
        c = input_train[i][:,3:]#rgb of noisyinput
        c=c[idxs]
        # idxs=torch.from_numpy(idxs)
        # idxs_a=idxs_a.squeeze(0)
        # idxs_b=torch.from_numpy(idxs)
        # idxs_b=idxs_b.squeeze(0)
        a=torch.from_numpy(a).int()
        b=torch.from_numpy(b).int()
#         a=np.ones((len(a),1),dtype=np.float32)
#         b=np.ones((len(b),1),dtype=np.float32)
        c=torch.from_numpy(c).float()
        d=torch.from_numpy(d).float()
#         locs.append(torch.cat([c,torch.LongTensor(c.shape[0],1).fill_(idx)],1))
        locs.append(torch.cat([b,torch.IntTensor(b.shape[0],1).fill_(idx)],1))
        feats.append(torch.cat([c,torch.FloatTensor(c.shape[0],1).fill_(idx)],1))
#         feats.append(torch.from_numpy(b))
#         groundlocs.append(torch.cat([d,torch.LongTensor(d.shape[0],1).fill_(idx)],1))
        groundlocs.append(torch.cat([a,torch.IntTensor(a.shape[0],1).fill_(idx)],1))
        groundfeats.append(torch.cat([d,torch.FloatTensor(d.shape[0],1).fill_(idx)],1))
        # groundfeats.append(torch.from_numpy(d))
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    groundlocs=torch.cat(groundlocs,0)
    groundfeats=torch.cat(groundfeats,0)
    # groundfeats=torch.cat(groundfeats,0)
    return {'x': [locs,feats], 'y': [groundlocs,groundfeats], 'id': tbl}
#     return locs, feats, groundlocs, groundfeats
val_data_loader = torch.utils.data.DataLoader(
    list(range(len(gt_val))),batch_size=batch_size, collate_fn=valMerge, num_workers=0,shuffle=True)
