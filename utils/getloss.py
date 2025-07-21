import os
import torch

####

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)

def checkpoint_save(model,exp_name,name2,epoch, use_cuda=True, delete=True):
    f=exp_name+'-%09d-'%epoch+name2+'.pth'
    model.cpu()
    torch.save(model.state_dict(),f)
    if use_cuda:
        model.cuda()
    #remove previous checkpoints unless they are a power of 2 to save disk space
    if delete:
        epoch=epoch-1
        f=exp_name+'-%09d-'%epoch+name2+'.pth'
        if os.path.isfile(f):
            if not is_power2(epoch):
                os.remove(f)

####

# A shape is (P_A, C), B shape is (P_B, C)
# D shape is (P_A, P_B)
def distance_matrix_general(A, B):
    r_A = torch.sum(A * A, dim=1, keepdim=True)
    r_B = torch.sum(B * B, dim=1, keepdim=True)
    m = torch.matmul(A.float(), torch.transpose(B, 1, 0).float())
    D = r_A - 2 * m + torch.transpose(r_B, 1, 0)
    return D

# gathers k points around queries. These are gathered from "points" using knn.
# points shape is (P_A, C), queries shape is (P_B, C)
# output shape is (P_B, k)
def gather_points_knn(points, queries, k, sort=True):
    D = distance_matrix_general(queries,points)    # (N, P_A, P_B)
    distances, point_indices = torch.topk(-D, k=k, sorted=sort)  # (N, P_A, K)
    return -distances, point_indices

####

# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_distance_matrix_general(A, B):
    r_A = torch.sum(A * A, dim=2, keepdim=True)
    r_B = torch.sum(B * B, dim=2, keepdim=True)
    m = torch.matmul(A.float(), torch.transpose(B, 2, 1).float())
    D = r_A - 2 * m + torch.transpose(r_B, 2, 1)
    return D

                
def knn_indices_general(queries, points, batchsize, k, sort=True):
#    point_num = queries.shape[1]
    D = distance_matrix_general(queries, points)
    distances, point_indices = torch.topk(-D, k=k, sorted=sort)  # (N, P, K)
#    batch_indices = tf.tile(tf.reshape(tf.range(batchsize), (-1, 1, 1, 1)), (1, point_num, k, 1))
#    indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
    return -distances

####

def get_loss(pred, noisyinput, ground):
    """
    pred: output in shape (N, 3)
    noisyinput: input of training in shape (N, 3)
    ground: ground truth of input in shape (M, 3)
    """
    batchsize = noisyinput.C[:, 0].max() + 1
    print(batchsize)
    
#    queries = output.view(batchsize,-1,3)
#    points = ground.view(batchsize,-1,3)
    
    loss_op = 0
    for i in range(batchsize):
        idx = noisyinput.C[:,0]==i
        inputxyz = noisyinput.F[idx].cuda()
        predict = pred.F[idx]   
        # Creating a one hot vector
        scalar = predict[:,0]
        scalar = scalar.unsqueeze(1)
        vector = predict[:,1:]
        _,targets = vector.max(dim=1)
        t = targets.unsqueeze(1)
        onehot = torch.zeros(vector.shape).cuda()
        onehot.scatter_(1, t, 1)
        proj = onehot*scalar


        output = inputxyz+proj
#     queries = output.unsqueeze(0)

#     idx = ground[:,3]==i
        gr = ground[idx]
#     points = gr.unsqueeze(0)

        distance = knn_indices_general(output, ground, 1, 1, True)
        loss_op += torch.mean(distance)

        distance = knn_indices_general(ground, output, 1, 1, True)
        loss_op += torch.mean(distance)
    
    return loss_op