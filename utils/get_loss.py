import torch

def get_loss(noisyinput, pred, ground):
    """
    noisyinput: tensor in shape (N, 3)
    pred: tensor in shape (N, 3)
    ground: tensor in shape (N, 3)
    """
#     batchsize = data2[0].max(0)[0][3] + 1
    
#    queries = output.view(batchsize,-1,3)
#    points = ground.view(batchsize,-1,3)
    
    loss_op = 0
#     for i in range(batchsize):
#     inputxyz = noisyinput[0][idx,:3].cuda()
#     predic = pred[idx]

    # Creating a one hot vector
    scalar = pred[:,0]
    vector = pred[:,1:]
    _,targets = vector.max(dim=1)
    t = targets.unsqueeze(1)
    onehot = torch.zeros(vector.shape).cuda()
    onehot.scatter_(1, t, 1)
    proj = onehot*scalar

    output = inputxyz+proj
    queries = output.unsqueeze(0)

    idx = ground[:,3]==i
    gr = ground[idx,:3]
    points = gr.unsqueeze(0)

    distance = knn_indices_general(queries, points, 1, 1, True)
    loss_op += torch.mean(distance)

    distance = knn_indices_general(points, queries, 1, 1, True)
    loss_op += torch.mean(distance)
    
    return loss_op
