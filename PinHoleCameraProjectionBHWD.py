#coding:utf-8
import torch
import torch.nn as nn  
from torch.autograd import Function
'''
PinHoleCameraProjectionBHWD will take b x h x w x 3 3D points and returns 
a projection of that on a 2D plabe as b x h x w x 2
for any 3D points repesented as (X, Y, Z) the projection is 

fx * ( X / Z ) + u0
fy * ( Y / Z ) + v0 

where fx and fy are focal lengths and (u0, v0) is camera cente
'''
class PinHoleCameraProjectionBHWD (Function):
    def __init__(self,height, width, fx, fy, u0, v0,t):
        super(PinHoleCameraProjectionBHWD,self).__init__()
        self.height = height
        self.width  = width
        self.t = t
        self.u0 = -1 + 2.0 * (u0-1)/(self.width-1) 
        self.v0 = -1 + 2.0 * (v0-1)/(self.height-1)
        
        #self.fx = fx
        #self.fy = fy

        self.u0 = self.u0 + ( 2.0 * t[0][0]/(self.width-1))
        self.v0 = self.v0 + ( 2.0 * t[0][1]/(self.height-1))
        
        self.fx =  (2.0 * fx*self.t[0][2]/(self.width-1))
        self.fy =  (2.0 * fy*self.t[0][3]/(self.height-1))

        '''
        what we want is this 
     	u' = -1 + ( fx*(X/Z) + u0 - 1) / (w - 1 ) * 2 
     	u' = -1 + 2 * (u0 - 1) / (w - 1) + 2*fx/(w-1) * (X/Z)
        '''
        self.epsilon = 1e-12
        self.output = torch.Tensor(1)
    def forward (self,_points3D):

        self.points3D = _points3D
        batchsize = self.points3D.size(0)
        
        self.output.resize_(self.points3D.size(0), self.points3D.size(1), self.points3D.size(2), 2).zero_()
        X = self.points3D.select(3,0) 
        Y = self.points3D.select(3,1) 
        Z = self.points3D.select(3,2) + self.epsilon
        '''
        u' = fx * (X/Z) + uo
        v' = fy * (Y/Z) + v0
        '''
        
        #self.new_u0 = self.u0 + ( 2.0 * t[0][0]/(self.width-1))
        #self.new_v0 = self.v0 + ( 2.0 * t[0][1]/(self.height-1))

        #self.new_fx =  (2.0 * self.fx*self.t[0][2]/(self.width-1))
        #self.new_fy =  (2.0 * self.fx*self.t[0][3]/(self.height-1))
        #x轴方向的平移和尺度变换
        self.output.select(3,0).copy_(torch.mul(torch.div(X,Z),self.fx) + self.u0 )
        #Y轴方向的平移和尺度变换
        self.output.select(3,1).copy_(torch.mul(torch.div(Y,Z),self.fy) + self.v0 )

        return  self.output

        
    
    def backward(self,_gradOut):
        
        self.gradOut = _gradOut
        self.gradInput = torch.Tensor(self.points3D.size()).zero_()

        self.grad_t = torch.zeros(1,4)
        
        X = self.points3D.select(3,0)
        Y = self.points3D.select(3,1)
        Z = self.points3D.select(3,2)
        Zs = self.points3D.select(3,2)

        X_div_Z = torch.div(-X,Z)
        Y_div_Z = torch.div(-Y,Z)
        
        Zs = Zs.add_(self.epsilon)    
        
        dLx = self.gradOut.select(3,0) 
        dLy = self.gradOut.select(3,1) 

        dLx_div_Z = torch.mul(torch.div(dLx,Zs),self.new_fx)  
        dLy_div_Z = torch.mul(torch.div(dLy,Zs),self.new_fy)

        self.gradInput.select(3,0).copy_(dLx_div_Z)
        self.gradInput.select(3,1).copy_(dLy_div_Z)
        
        self.gradInput.select(3,2).copy_(torch.mul(dLx_div_Z,X_div_Z) + torch.mul(dLy_div_Z,Y_div_Z))
        #下面的代码有错
        '''
        dLtrans_x = self.u0 + 2./(self.width - 1)
        
        dLtrans_y = self.v0 + 2./(self.height - 1)
        dLscale_x = 2 * (dLx/Zs) * self.fx/(self.width - 1)
        dLscale_y = 2 * (dLy/Zs) * self.fy/(self.height - 1)
        self.grad_t[0][0] = dLtrans_x
        self.grad_t[0][1] = dLtrans_y
        self.grad_t[0][2] = dLscale_x.sum()
        self.grad_t[0][3] = dLscale_y.sum()
        #print(self.grad_t)
        '''
        return self.gradInput,self.grad_t
    