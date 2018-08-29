#coding:utf-8
import sys
import torch
import torch.nn as nn
from torch.autograd import Function
from TransformationRotationSO3 import TransformationRotationSO3
from Transform3DPoints_R import Transform3DPoints_R
from PinHoleCameraProjectionBHWD import PinHoleCameraProjectionBHWD
from ReverseXYOrder import ReverseXYOrder
from BilinearSamplerBHWD import BilinearSamplerBHWD
height = 240
width  = 320
u0     = 160
v0     = 120
fx = 240
fy = 240

t = torch.zeros(1,4)
t[0][0] = 0.    #x平移
t[0][1] = 0.
t[0][2] = 1.    #x尺度变换
t[0][3] = 1.

class rotation_net(Function):
    def __init__ (self):
        super(rotation_net,self).__init__()
        self.transformationRotationSO3 = TransformationRotationSO3()

        self.transform3DPoints_R = Transform3DPoints_R(height, width, fx, fy, u0, v0)
        self.PinHoleCameraProjectionBHWD = PinHoleCameraProjectionBHWD(height, width, fx, fy, u0, v0,t)
        self.reversexyorder = ReverseXYOrder(1,height,width)
    def forward(self,x):

        x = self.transformationRotationSO3(x)
        x = self.transform3DPoints_R(x)
        x = self.PinHoleCameraProjectionBHWD(x)
        x = self.reversexyorder(x)

        return x

    def backward(self,x,gradOutput):
        x = x
        self.forward(x)
        x = self.reversexyorder.backward(gradOutput)
        x = self.PinHoleCameraProjectionBHWD.backward(x)
        x = self.transform3DPoints_R.backward(x)
        x = self.transformationRotationSO3.backward(x)
        return x