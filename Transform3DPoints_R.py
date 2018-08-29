#coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Function
class Transform3DPoints_R(Function):
    def __init__(self,height, width, fx, fy, u0, v0):
        super(Transform3DPoints_R,self).__init__()
        
        self.height = height
        self.width = width
        self.u0 = u0  #-1 + (u0-1)/(self.width-1) * 2
        self.v0 = v0  #-1 + (v0-1)/(self.height-1) * 2
        self.fx = fx  #2 * fx/(self.width-1)
        self.fy = fy  #2 * fy/(self.height-1)
        self.scale = 1.0
        self.baseGrid = torch.zeros(height, width, 3)
        
        for i in range(self.height): 
            self.baseGrid.select(2,1).select(0,i).fill_(self.scale * (i + 1 -self.v0)/self.fy)
        
        for j in range(self.width): 
            self.baseGrid.select(2,0).select(1,j).fill_(self.scale * (j + 1 -self.u0)/self.fx)
        
        self.baseGrid.select(2,2).fill_(1)
        self.batchGrid = torch.Tensor(1, height, width, 3).copy_(self.baseGrid)
        
    def forward(self,_transformMatrix):
        self.transformParams = _transformMatrix
        batchsize = self.transformParams.size(0)
        if self.batchGrid.size(0) != batchsize :
            self.batchGrid.resize(batchsize, self.height, self.width, 3)
            for i in range (batchsize): 
                self.batchGrid.select(0,i).copy_(self.baseGrid)
        
        self.output = torch.zeros(batchsize, self.height*self.width, 3)
        flattenedBatchGrid = self.batchGrid.view(batchsize, self.width*self.height, 3)
        #flattenedOutput = self.output.view(batchsize, self.width*self.height, 3)
        self.output = torch.bmm(flattenedBatchGrid,_transformMatrix.transpose(1,2))
        self.output.resize_(batchsize, self.height,self.width, 3)
        
        
        return self.output 

    
    def backward (self,_gradGrid):
        
        gradGrid = _gradGrid
        batchsize = self.transformParams.size(0)
        flattenedGradGrid = gradGrid.view(batchsize,self.width*self.height,3)
        flattenedBatchGrid = self.batchGrid.view(batchsize, self.width*self.height, 3)
        self.gradInput = torch.Tensor(self.transformParams.size()).zero_()
        self.gradInput = self.gradInput.baddbmm(flattenedGradGrid.transpose(1,2), flattenedBatchGrid)
        
        return self.gradInput
        