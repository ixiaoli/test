import torch
import torch.nn as nn
from torch.autograd import Function,Variable
import torch.autograd.function
class ReverseXYOrder (Function):
    def __init__(self,batchsize,height,width):
        super(ReverseXYOrder,self).__init__()
        self.batchsize = batchsize
        self.height = height
        self.width  = width
    
    def forward(self,input):
        self.input =input
        
        self.output = torch.zeros(self.batchsize, self.height,self.width, 2)
        self.output.select(3,0).copy_(self.input.select(3,1))
        self.output.select(3,1).copy_(self.input.select(3,0))

        return self.output 
    
    def backward(self,gradOutput):
        
        self.gradInput = gradOutput
        self.gradInput = torch.Tensor(self.input.size()).zero_()                                                                       
        self.gradInput.select(3,0).copy_(gradOutput.select(3,1))
        self.gradInput.select(3,1).copy_(gradOutput.select(3,0))
        #print(self.gradInput)
        return self.gradInput
        
