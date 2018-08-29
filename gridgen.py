#coding:utf-8
import sys
sys.path.append("..")
from torch.nn.modules.module import Module
import torch
from torch.autograd import Variable
import numpy as np
from Functions.TransformationRotationSO3 import TransformationRotationSO3
from Functions.Transform3DPoints_R import Transform3DPoints_R
from Functions.PinHoleCameraProjectionBHWD import PinHoleCameraProjectionBHWD
from Functions.ReverseXYOrder import ReverseXYOrder
from Functions.BilinearSamplerBHWD import BilinearSamplerBHWD

height = 240
width = 320
fx = 240
fy = 240
u0 = 120
v0 = 160
batchsize = 1


t = torch.Tensor(1,4).zero_()
t[0][0] = 0.    #x平移
t[0][1] = 0.
t[0][2] = 1.    #x尺度变换
t[0][3] = 1.


class SO3GridGen(Module):
    def __init__(self,height,width,lr=1):
        super(SO3GridGen,self).__init__()

        self.B = BilinearSamplerBHWD()
        self.height = height
        self.width = width
        self.lr = lr

    def forward(self,input,input1):
        
        SO3Matrix = TransformationRotationSO3()(input)
        threeDimGrid = Transform3DPoints_R(height,width,fx,fy,u0,v0)(SO3Matrix)
        projection2D = PinHoleCameraProjectionBHWD(height,width,fx,fy,u0,v0,t)(threeDimGrid)
        grid = ReverseXYOrder(batchsize,height,width)(projection2D)
        
        
        return self.B(input1,grid)
