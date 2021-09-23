#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import os

import numpy as np

import torch
import torchvision
import torchsummary
from torchsummary import summary


########################################### 가중치 초기화 하는 구간 START ##############################################

def weights_init_normal(m): #경우에 따라서 아래의 값들로 가중치를 초기화 하겠다는 말임
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: #비교연산자 다르면 True값을 연산 
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
'''
https://yngie-c.github.io/deep%20learning/2020/03/17/parameter_init/
Zero initialization
가장 먼저 떠오르는 생각은 “모든 파라미터 값을 0으로 놓고 시작하면 되지 않을까?” 입니다. 하지만 이는 너무나도 단순한 생각입니다. 
좀 더 자세히 말하자면 신경망의 파라미터가 모두 같아서는 안됩니다.

파라미터의 값이 모두 같다면 역전파(Back propagation)를 통해서 갱신하더라도 모두 같은 값으로 변하게됩니다. 
신경망 노드의 파라미터가 모두 동일하다면 여러 개의 노드로 신경망을 구성하는 의미가 사라집니다. 결과적으로 층마다 한 개의 노드만을 배치하는 것과 같기 때문이지요. 그래서 초깃값은 무작위로 설정해야 합니다.

Random Initialization
파라미터에 다른 값을 부여하기 위해서 가장 쉽게 생각해 볼 수 있는 방법은 확률분포를 사용하는 것이지요. 정규분포를 이루는 값을 각 가중치에 배정하여 모두 다르게 설정할 수 있습니다. 표준편차를 다르게 설정하면서 정규분포로 가중치를 초기화한 신경망의 활성화 함수 출력 값을 시각화해보겠습니다. 신경망은 100개의 노드를 5층으로 쌓았습니다.

먼저 표준편차가 1인 케이스를 알아보겠습니다. 활성화 함수로는 시그모이드(로지스틱) 함수를 사용하였습니다.



'''


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'): # 가중치를 초기화 시켜준다는 것 같음
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

########################################### 가중치 초기화 하는 구간 END ##############################################


class FeatureExtraction(nn.Module):
    def __init__(self, input_nc, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FeatureExtraction, self).__init__()
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1) #ngf 다음 node숫자
        model = [downconv, nn.ReLU(True), norm_layer(ngf)]
        #BatchNorm2d 얘가 인풋의 정규분포대로 아웃풋의 정규분포가 나올수 있게 가중치들의 수치를 조절해주는것 즉 가중치가 한쪽으로 너무 치우치지 않게 해주는것 
        for i in range(n_layers):
            in_ngf = 2**i * ngf if 2**i * ngf < 512 else 512
            out_ngf = 2**(i+1) * ngf if 2**i * ngf < 512 else 512
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, nn.ReLU(True)]
            model += [norm_layer(out_ngf)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        model += [norm_layer(512)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        self.model = nn.Sequential(*model)
        init_weights(self.model, init_type='normal')  #여기 이코드는 모델을 불러올 때 마다 가중치를 랜덤값으로 설정하는데 그 초기 가중치 값을 위의 코드를 통해서 어느정도
        #우리가 설정해주겠다는 것임

    def forward(self, x):
        
        return self.model(x)


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        # norm = torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        #unsqueeze(1) 해당 숫자의 차원에 1을 넣어 차원생성 (12,10)-> (12,1,10)
        #expand_as(x) x의 텐서 차원숫자만큼 확장시키겠다 부족한 값들은 그대로 복사
        # -> ? torch.Size([4, 16, 12])
        # print(norm)
        '''
        원래 -1~1사이의 값들이 개커짐  그걸 expand_as로 1차원에 512장만 복사하는것
         [[ 8.0053, 10.5621, 13.2573, 18.5879, 22.8604, 31.2243, 33.5784,
          32.1775, 25.4193, 16.7022, 13.4100,  8.1757],
         [ 9.9400, 13.2302, 16.0496, 22.6015, 30.2104, 33.4168, 35.4299,
        '''
        '''
        즉 위의것은 내부의 값들을 제곱하고, 512장으로 쪼개진 사진들을 모두 더해버리고 루트값을 씌어 버린다음에 없어진 1차원을 다시 만들고
        거기를 원래의 모양으로 되돌리기 위해서 값들을 복사한다는것
        '''
        # print(norm)
        # print('노말라이즈를 어캐 한다는 겨?',norm.shape)
        #노말라이즈를 어캐 한다는 겨? torch.Size([4, 512, 16, 12])
        #노말라이즈를 어캐 한다는 겨? torch.Size([4, 512, 16, 12])

        

        '''
        x2 = torch.pow(x1,2) 요소의 제곱을 의미한다.
        print(x2)

        tensor([[ 1.,  4.,  9.],
        [16., 25., 36.]])

        즉 torch.pow(x1,0.5)는 루트를 씌어준것과 같다. 

        torch.sum(x, 1 ) 가로로 쭉 더해준다. 
        아래 참조
        https://velog.io/@reversesky/torch.sum%EC%97%90-%EB%8C%80%ED%95%B4-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90

        '''
        return torch.div(feature,norm)
    
class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        print('feature_A',feature_A.size()) #([4, 512, 16, 12])
        print('feature_B',feature_B.size()) #([4, 512, 16, 12])


        '''
        텐서형태로 이미지를 바꾸면 베치, 채널, 높이, 넓이가 되는 이유는 몇개의 배치 묵음으로 된 몇개의 채널 요소를 가진 높이, 넓이 얼마의 자료다라는 의미 
        '''
        # reshape features for matrix multiplication
        #([4, 512, 16, 12])
        # print(feature_A)
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)
        #b,h*w,c

        '''
        congiguous()의 용도
        arrow(), view(), expand(), transpose() 등의 함수는 새로운 Tensor를 생성하는 게 아니라 
        기존의 Tensor에서 메타데이터만 수정하여 우리에게 정보를 제공합니다. 즉 메모리상에서는 같은 공간을 공유합니다.

        하지만 연산 과정에서 Tensor가 메모리에 올려진 순서(메모리 상의 연속성)가 중요하다면 원하는 결과가 나오지 않을 수 있고
        에러가 발생합니다. 그렇기에 어떤 함수 결과가 실제로 메모리에도 우리가 기대하는 순서로 유지하려면 
        contiguous()를 사용하여 에러가 발생하는 것을 방지할 수 있습니다.
        
        '''
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)
        print(feature_mul.size()) #torch.Size([4, 192, 192])
        # print(feature_mul)


        # ==> 행렬의 곱을 계산하여 ==> b ,h*w,h*w
        #torch.bmm 정확히 사이즈가 일치하지 않아도 곱하기 가능
        '''
        >>> input = torch.randn(10, 3, 4)
        >>> mat2 = torch.randn(10, 4, 5)
        >>> res = torch.bmm(input, mat2)
        >>> res.size()
        torch.Size([10, 3, 5])
        '''
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        # b,h*w,h,w
        return correlation_tensor
    
class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512,output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 4 * 3, output_dim)
        self.tanh = nn.Tanh()
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()
            self.tanh.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        x = self.tanh(x)
        return x
class AffineGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, out_ch = 3):
        super(AffineGridGen, self).__init__()        
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        
    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size,self.out_ch,self.out_h,self.out_w))
        return F.affine_grid(theta, out_size)
        
class TpsGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        #meshgrid 점을 받아서 격자를 만든다.
        #linspace(1,10, num) 1~10사이의 num개의 균일한 숫자를 만들어낸다.
        # grid_X,grid_Y: size [1,H,W,1,1]
        # print('gridx는 뭐야',self.grid_X.shape)gridx는 뭐야 (256, 192)
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        #FloatTensor 소수점을 가지는 텐서로 바꾼다
        #0차원 3차원 추가함


        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1,1,grid_size)
            # print(grid_size) #5
            # print(axis_coords) [-1.  -0.5  0.   0.5  1. ]
            self.N = grid_size*grid_size
            P_Y,P_X = np.meshgrid(axis_coords,axis_coords)
            # print(P_X)
            '''
            print(P_X)
            [[-1.  -1.  -1.  -1.  -1. ]
            [-0.5 -0.5 -0.5 -0.5 -0.5]
            [ 0.   0.   0.   0.   0. ]
            [ 0.5  0.5  0.5  0.5  0.5]
            [ 1.   1.   1.   1.   1. ]]
            
            
            print(P_Y)

            [[-1.  -0.5  0.   0.5  1. ]
            [-1.  -0.5  0.   0.5  1. ]
            [-1.  -0.5  0.   0.5  1. ]
            [-1.  -0.5  0.   0.5  1. ]
            [-1.  -0.5  0.   0.5  1. ]]
            '''
            # print(P_Y.shape) (5, 5)

            P_X = np.reshape(P_X,(-1,1)) # size (25,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (25,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            #clone() : 기존 Tensor와 내용을 복사한 텐서 생성

            self.Li = self.compute_L_inverse(P_X,P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()
                self.P_X_base = self.P_X_base.cuda()
                self.P_Y_base = self.P_Y_base.cuda()

            
    def forward(self, theta):
        warped_grid = self.apply_transformation(theta,torch.cat((self.grid_X,self.grid_Y),3))
        
        return warped_grid
    
    def compute_L_inverse(self,X,Y):
        N = X.size()[0] # num of points (along dim 0)
        #25를 가져오겠지 ? X가 (25,1)이니깐
        # construct matrix K
        # print('x의 사이즈',X.size()) x의 사이즈 torch.Size([25, 1])
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        # print('xmat의 사이즈',Xmat.size()) xmat의 사이즈 torch.Size([25, 25])
        # print(Xmat)
        # print(Xmat.transpose(0,1)) #큐브를 옆으로 돌려버렸네

        '''
        ex)
        -1-1-1         1 0-1
         0 0 0   ->    1 0-1
         1 1 1         1 0-1

        '''
        # print(torch.pow(Xmat-Xmat.transpose(0,1),2))

        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        # print('이거 사이즈는 ??',P_dist_squared.shape) ([25, 25])
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        
        # construct matrix L
        O = torch.FloatTensor(N,1).fill_(1)
        
        Z = torch.FloatTensor(3,3).fill_(0)       
        P = torch.cat((O,X,Y),1)
        

        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L)
        # print('Li는 뭐지??',Li)
        if self.use_cuda:
            Li = Li.cuda()
        return Li
        
    def apply_transformation(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords  
        # and points[:,:,:,1] are the Y coords  
        
        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X=theta[:,:self.N,:,:].squeeze(3)
        Q_Y=theta[:,self.N:,:,:].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)
        
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N))
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N))
        
        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        
        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])
        
        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
                    
        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)
        
        return torch.cat((points_X_prime,points_Y_prime),3)
        
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class GMM(nn.Module):
    """ Geometric Matching Module
    """
    def __init__(self, opt):
        super(GMM, self).__init__()
        self.extractionA = FeatureExtraction(22, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d) 
        self.extractionB = FeatureExtraction(3, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=192, output_dim=2*opt.grid_size**2, use_cuda=True)
        self.gridGen = TpsGridGen(opt.fine_height, opt.fine_width, use_cuda=True, grid_size=opt.grid_size)
        
    def forward(self, inputA, inputB):
        # print('인풋A는',inputA.size())
        featureA = self.extractionA(inputA)
        featureB = self.extractionB(inputB)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        correlation = self.correlation(featureA, featureB)

        theta = self.regression(correlation)
        grid = self.gridGen(theta)
        return grid, theta

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()