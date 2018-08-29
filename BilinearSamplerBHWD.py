#-*- coding:utf-8 -*-
import sys
sys.path.append("..")
import torch
from cffi import FFI
from _ext import my_lib
from torchvision import transforms
from torch.autograd import Function

class BilinearSamplerBHWD(Function):

    def forward(self,input1,input2):
        self.input1 = input1
        self.input2 = input2
        inputImages = self.input1
        grids = self.input2
        self.output = torch.zeros(input1.size()[0], input1.size()[1], input1.size()[2], input1.size()[3])
        my_lib.BilinearSamplerBHWD_updateOutput(inputImages, grids, self.output) 
        
        return self.output

    def backward (self,_gradOutput):
        #print('start backward')
        gradInputImages = torch.zeros(self.input1.size())
        gradGrids = torch.zeros(self.input2.size())
        inputImages = self.input1
        grids = self.input2

        gradOutput = _gradOutput
        
        my_lib.BilinearSamplerBHWD_updateGradInput(inputImages, grids, gradInputImages, gradGrids, gradOutput)
        #print(gradInputImages,gradGrids)
        return gradInputImages,gradGrids
        

    
