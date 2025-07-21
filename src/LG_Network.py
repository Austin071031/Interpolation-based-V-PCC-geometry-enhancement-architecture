import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import torch.nn.functional as F
import torchsummary
import numpy as np

from knn_cuda import KNN
from pointnet2.pointnet2_utils import gather_operation,grouping_operation
from model.BasicBlock import MyInception_1, Pyramid_1

class get_edge_feature(nn.Module):
    """construct edge feature for each point
    Args:
        tensor: input a point cloud tensor,batch_size,num_dims,num_points
        k: int
    Returns:
        edge features: (batch_size,num_dims,num_points,k)
    """
    def __init__(self,k=16):
        super(get_edge_feature,self).__init__()
        self.KNN=KNN(k=k+1,transpose_mode=False)
        self.k=k
    def forward(self,point_cloud):
        dist,idx=self.KNN(point_cloud,point_cloud)
        '''
        idx is batch_size,k,n_points
        point_cloud is batch_size,n_dims,n_points
        point_cloud_neightbors is batch_size,n_dims,k,n_points
        '''
        idx=idx[:,1:,:]
        point_cloud_neighbors=grouping_operation(point_cloud,idx.contiguous().int())
        point_cloud_central=point_cloud.unsqueeze(2).repeat(1,1,self.k,1)
        #print(point_cloud_central.shape,point_cloud_neighbors.shape)
        edge_feature=torch.cat([point_cloud_central,point_cloud_neighbors-point_cloud_central],dim=1)

        return edge_feature,idx

        return dist,idx


    
class denseconv(ME.MinkowskiNetwork):
    def __init__(self, growth_rate=64, k=16, in_channels=3, bn_momentum=0.1, D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        
#         self.edge_feature_model=get_edge_feature(k=k)
        
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=growth_rate,
            kernel_size=1,
            stride=1,
            dimension=D)
        self.norm1 = ME.MinkowskiBatchNorm(growth_rate, momentum=bn_momentum)
        
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=growth_rate+in_channels,
            out_channels=growth_rate,
            kernel_size=1,
            stride=1,
            dimension=D)
        self.norm2 = ME.MinkowskiBatchNorm(growth_rate, momentum=bn_momentum)
        
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=2*growth_rate+in_channels,
            out_channels=growth_rate,
            kernel_size=1,
            stride=1,
            dimension=D)
        self.norm3 = ME.MinkowskiBatchNorm(growth_rate, momentum=bn_momentum)
        
        
    def forward(self, inputs):
        '''
        y should be batch_size,in_channel,k,n_points
        '''
#         y,idx=self.edge_feature_model(inputs)
        
        conv1 = self.conv1(inputs)
        conv1 = self.norm1(conv1)
        conv1 = MEF.relu(conv1)
        inter_result = ME.cat(conv1, inputs)#concat on feature dimension
        
        conv2 = self.conv2(inter_result)
        conv2 = self.norm2(conv2)
        conv2 = MEF.relu(conv2)
        inter_result = ME.cat(conv2, inter_result)
        
        conv3 = self.conv3(inter_result)
        conv3 = self.norm3(conv3)
        inter_result = ME.cat(conv3, inter_result)
        
        
#         pooled_F, _ = torch.topk(inter_result.F, k=3, dim=1, largest=True)

#         final_result = ME.SparseTensor(features=pooled_F, coordinates=inter_result.C)
        
        return inter_result
        
class feature_extraction(ME.MinkowskiNetwork):
    def __init__(self, bn_momentum=0.1, D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        
        self.growth_rate = 24
        self.dense_n = 3
        self.knn = 16
        self.input_channel = 3
        comp = self.growth_rate*2
        '''
        input of conv1 is batch_size,num_dims,num_points
        '''
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=self.input_channel,
            out_channels=24,
            kernel_size=1,
            stride=1,
            dimension=D)
        self.norm1 = ME.MinkowskiBatchNorm(24, momentum=bn_momentum)
        self.denseconv1 = denseconv(in_channels=24, growth_rate=self.growth_rate)
        
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=120,
            out_channels=comp,
            kernel_size=1,
            stride=1,
            dimension=D)
        self.norm2 = ME.MinkowskiBatchNorm(comp, momentum=bn_momentum)
        self.denseconv2 = denseconv(in_channels=comp, growth_rate=self.growth_rate)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=240,
            out_channels=comp,
            kernel_size=1,
            stride=1,
            dimension=D)
        self.norm3 = ME.MinkowskiBatchNorm(comp, momentum=bn_momentum)
        self.denseconv3 = denseconv(in_channels=comp, growth_rate=self.growth_rate)
        
        self.conv4 = ME.MinkowskiConvolution(
            in_channels=360,
            out_channels=comp,
            kernel_size=1,
            stride=1,
            dimension=D)
        self.norm4 = ME.MinkowskiBatchNorm(comp, momentum=bn_momentum)
        self.denseconv4 = denseconv(in_channels=comp, growth_rate=self.growth_rate)
        
    def forward(self, inputs):
        l0_features = self.conv1(inputs) #n, 24
        l0_features = self.norm1(l0_features)
        l0_features = MEF.relu(l0_features)
        l1_features = self.denseconv1(l0_features) #n,24+24*3=96
        l1_features = ME.cat(l1_features, l0_features) #n,96+24=120
        
        l2_features = self.conv2(l1_features) #n, 48
        l2_features = self.norm2(l2_features)
        l2_features = MEF.relu(l2_features)
        l2_features = self.denseconv2(l2_features) #b,48+24*3=120,n
        l2_features = ME.cat(l2_features, l1_features) #b,120+120=240,n
        
        l3_features = self.conv3(l2_features) #b, 48, n
        l3_features = self.norm3(l3_features)
        l3_features = MEF.relu(l3_features)
        l3_features = self.denseconv3(l3_features) #n,48+24*3=120
        l3_features = ME.cat(l3_features, l2_features) #n,120+240=360
        
        l4_features = self.conv4(l3_features) #n, 48
        l4_features = self.norm4(l4_features)
        l4_features = MEF.relu(l4_features)
        l4_features = self.denseconv4(l4_features) #n,48+24*3=120
        l4_features = ME.cat(l4_features, l3_features) #n,120+360=480
        
        return l4_features
    
class down_projection_unit(ME.MinkowskiNetwork):
    BLOCK_1 = MyInception_1
    BLOCK_2 = Pyramid_1
    
    def __init__(self, in_channels=480, out_channels=3, bn_momentum=0.1, D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        BLOCK_1 = self.BLOCK_1
        BLOCK_2 = self.BLOCK_2
        
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm0 = ME.MinkowskiBatchNorm(128, momentum=bn_momentum)
#         self.attention0 = attention_unit(in_channels=128)
        self.block0 = self.make_layer(BLOCK_1, BLOCK_2, 128, bn_momentum=bn_momentum, D=D)
        
        ### downsample
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm1 = ME.MinkowskiBatchNorm(256, momentum=bn_momentum)
        self.maxpooling1 = ME.MinkowskiMaxPooling(kernel_size=3, stride=3, dimension=D)
#         self.attention1 = attention_unit(in_channels=256)
        self.block1 = self.make_layer(BLOCK_1, BLOCK_2, 256, bn_momentum=bn_momentum, D=D)
        
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2 = ME.MinkowskiBatchNorm(512, momentum=bn_momentum)
        self.maxpooling2 = ME.MinkowskiMaxPooling(kernel_size=3, stride=3, dimension=D)
#         self.attention2 = attention_unit(in_channels=512)
        self.block2 = self.make_layer(BLOCK_1, BLOCK_2, 512, bn_momentum=bn_momentum, D=D)
        
#         self.conv3 = ME.MinkowskiConvolution(
#             in_channels=512,
#             out_channels=1024,
#             kernel_size=3,
#             stride=2,
#             dilation=1,
#             bias=False,
#             dimension=D)
#         self.norm3 = ME.MinkowskiBatchNorm(1024, momentum=bn_momentum)
# #         self.attention3 = attention_unit(in_channels=256)
#         self.block3 = self.make_layer(BLOCK_1, BLOCK_2, 1024, bn_momentum=bn_momentum, D=D)
        
        ### upsample
        self.conv3_tr = ME.MinkowskiConvolution(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3_tr = ME.MinkowskiBatchNorm(256, momentum=bn_momentum)
        self.pooling3_tr = ME.MinkowskiPoolingTranspose(kernel_size=3, stride=3, dimension=D)
#         self.attention4_tr = attention_unit(in_channels=256)
        self.block3_tr = self.make_layer(BLOCK_1, BLOCK_2, 256, bn_momentum=bn_momentum, D=D)
        
        self.conv2_tr = ME.MinkowskiConvolution(
          in_channels=256+256,
          out_channels=128,
          kernel_size=3,
          stride=1,
          dilation=1,
          bias=False,
          dimension=D)
        self.norm2_tr = ME.MinkowskiBatchNorm(128, momentum=bn_momentum)
        self.pooling2_tr = ME.MinkowskiPoolingTranspose(kernel_size=3, stride=3, dimension=D)
#         self.attention3_tr = attention_unit(in_channels=512)
        self.block2_tr = self.make_layer(BLOCK_1, BLOCK_2, 128, bn_momentum=bn_momentum, D=D)
        
#         self.conv1_tr = ME.MinkowskiConvolutionTranspose(
#           in_channels=256,
#           out_channels=128,
#           kernel_size=3,
#           stride=2,
#           dilation=1,
#           bias=False,
#           dimension=D)
#         self.norm1_tr = ME.MinkowskiBatchNorm(128, momentum=bn_momentum)
# #         self.attention2_tr = attention_unit(in_channels=256)
#         self.block1_tr = self.make_layer(BLOCK_1, BLOCK_2, 128, bn_momentum=bn_momentum, D=D)
        
        
    def make_layer(self, block_1, block_2, channels, bn_momentum, D):
        layers = []
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))
        layers.append(block_2(channels=channels, bn_momentum=bn_momentum))
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))

        return nn.Sequential(*layers)
        
    def forward(self, inputs):
        l0 = self.conv0(inputs)
        l0 = self.norm0(l0)
        l0 = self.block0(l0)
        l0 = MEF.relu(l0)
#         print(l0.F.shape)
        
        #downsample
        out_d1 = self.conv1(l0)
        out_d1 = self.norm1(out_d1)
        out_d1 = self.maxpooling1(out_d1)
#         print(d1.F.shape)
        out_d1 = self.block1(out_d1)
        d1 = MEF.relu(out_d1)
        
        out_d2 = self.conv2(d1)
        out_d2 = self.norm2(out_d2)
        out_d2 = self.maxpooling2(out_d2)
#         print(d2.F.shape)
        out_d2 = self.block2(out_d2)
        d2 = MEF.relu(out_d2)
        
        #upsample
        u2 = self.conv3_tr(d2)
        u2 = self.norm3_tr(u2)
        u2 = self.pooling3_tr(u2)
#         print(u3.F.shape)
        u2 = self.block3_tr(u2)
        out_u2 = MEF.relu(u2)

        out_u3 = ME.cat(out_u2, out_d1)
        
        u1 = self.conv2_tr(out_u1)
        u1 = self.norm2_tr(u1)
        u1 = self.pooling2_tr(u1)
#         print(u2.F.shape)
        u1 = self.block2_tr(u1)
        out_u1 = MEF.relu(u1)
        
        #self correction
        delta = u1 - l0
        
        delta1 = self.conv1(delta)
        delta1 = self.norm1(delta1)
        delta1 = self.maxpooling1(delta1)
        delta1 = self.block1(delta1)
        delta1 = MEF.relu(delta1)
        
        delta2 = self.conv2(delta1)
        delta2 = self.norm2(delta2)
        delta2 = self.maxpooling2(delta2)
        delta2 = self.block2(delta2)
        delta2 = MEF.relu(delta2)
                
        self_correct_down = d2 + delta2
        
        return self_correct_down
    
    
class Generator(ME.MinkowskiNetwork):
    def __init__(self, in_channels=3, out_channels=3, bn_momentum=0.1, D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        
        self.feature_extractor = feature_extraction()
        self.down_projection_unit = down_projection_unit()
        
        self.final1 = ME.MinkowskiConvolution(
            in_channels=512,
            out_channels=256,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.final1_bn = ME.MinkowskiBatchNorm(256, momentum=bn_momentum)
        
        self.final2 = ME.MinkowskiConvolution(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.final2_bn = ME.MinkowskiBatchNorm(128, momentum=bn_momentum)
        
        self.final3 = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=64,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.final3_bn = ME.MinkowskiBatchNorm(64, momentum=bn_momentum)
        
        self.final4 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        
    def forward(self, inputs):
        features = self.feature_extractor(inputs) #n, 480
        
        H = self.down_projection_unit(features) #128,n/x
        
        coord = self.final1(H)
        coord = self.final1_bn(coord)
        coord = MEF.relu(coord)
        
        coord = self.final2(coord)
        coord = self.final2_bn(coord)
        coord = MEF.relu(coord)
        
        coord = self.final3(coord)
        coord = self.final3_bn(coord)
        coord = MEF.relu(coord)
        
        coord = self.final4(coord)
        
        return coord
        
        
# if __name__=="__main__":
#     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     generator=Generator(out_channels=3).cuda()
#     feats=torch.randint(0,1025, (32768,3)).cuda()
#     coord=torch.randint(0,255, (32768,3)).cuda()
#     b = torch.zeros(coord.shape[0], dtype=int)
#     b = b.unsqueeze(1).cuda()
#     coord = torch.cat((b, coord), 1)
#     x = ME.SparseTensor(features=feats.float(), coordinates=coord.int())
    
#     Total_params = 0
#     Trainable_params = 0
#     NonTrainable_params = 0

#     # 遍历model.parameters()返回的全局参数列表
#     for param in generator.parameters():
#         mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
#         Total_params += mulValue  # 总参数量
#         if param.requires_grad:
#             Trainable_params += mulValue  # 可训练参数量
#         else:
#             NonTrainable_params += mulValue  # 非可训练参数量

#     print(f'Total params: {Total_params}')
#     print(f'Trainable params: {Trainable_params}')
#     print(f'Non-trainable params: {NonTrainable_params}')
# #     output=generator(x)
# #     print(output.F.shape)       
        