#coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Function


class TransformationRotationSO3(Function):
    def __init__(self):
        super(TransformationRotationSO3,self).__init__()
        self.threshold = 1e-12 

    def dR_by_dvi(self,transparams, RotMats, which_vi, threshold):
    
        omega_x = transparams.select(1,0)
        
        omega_y = transparams.select(1,1)
        
        omega_z = transparams.select(1,2)
        omega_skew = torch.Tensor(RotMats.size()).type_as(transparams)
        omega_skew.zero_()

        omega_skew.select(2,0).select(1,1).copy_(omega_z)
        omega_skew.select(2,0).select(1,2).copy_(-omega_y)
        omega_skew.select(2,1).select(1,0).copy_(-omega_z)
        omega_skew.select(2,1).select(1,2).copy_(omega_x)

        omega_skew.select(2,2).select(1,0).copy_(omega_y)
        omega_skew.select(2,2).select(1,1).copy_(-omega_x)
        #print('omega_skew',omega_skew)
        Id_minus_R_ei = torch.Tensor(RotMats.size(0),RotMats.size(1),1).zero_().type_as(transparams) 
        Id_minus_R_ei.select(1,which_vi).add_(1)	 
        #print('Id_minus_R_ei',Id_minus_R_ei)

        I = torch.Tensor(RotMats.size(0), RotMats.size(1), RotMats.size(2)).zero_().type_as(transparams)
        I.select(1,0).select(1,0).add_(1)
        I.select(1,1).select(1,1).add_(1)
        I.select(1,2).select(1,2).add_(1)
        
        Id_minus_R_ei = torch.bmm(torch.add(I,-RotMats), Id_minus_R_ei)

        v_cross_Id_minus_R_ei = torch.bmm(omega_skew,Id_minus_R_ei)
        cross_x = v_cross_Id_minus_R_ei.select(1,0)	
        cross_y = v_cross_Id_minus_R_ei.select(1,1)	
        cross_z = v_cross_Id_minus_R_ei.select(1,2)	
        vcross = torch.Tensor(RotMats.size()).type_as(transparams)
        vcross.zero_()
        vcross.select(2,0).select(1,1).copy_(cross_z.view_as(vcross.select(2,0).select(1,1)))
        vcross.select(2,0).select(1,2).copy_(-cross_y.view_as(vcross.select(2,0).select(1,2)))
        vcross.select(2,1).select(1,0).copy_(-cross_z.view_as(vcross.select(2,1).select(1,0)))
        vcross.select(2,1).select(1,2).copy_(cross_x.view_as(vcross.select(2,1).select(1,2)))
        vcross.select(2,2).select(1,0).copy_(cross_y.view_as(vcross.select(2,2).select(1,0)))
        vcross.select(2,2).select(1,1).copy_(-cross_x.view_as(vcross.select(2,2).select(1,1)))
        
        omega_mag = torch.pow(omega_x,2) + torch.pow(omega_y,2) + torch.pow(omega_z,2)
        omega_selected = transparams.select(1,which_vi)
        
        for b in range(omega_skew.size(0)):
            if omega_mag[b] > threshold :
                #print('lll')
                v_i = omega_selected[b]
                omega_skew[b] = omega_skew[b].mul(v_i) + vcross[b]
                omega_skew[b] = omega_skew[b].div(omega_mag[b])
            else :
                
                e_i = torch.Tensor(3,1).type_as(transparams).zero_()
                e_i.select(1,which_vi).fill_(1)
                eMat = torch.Tensor(3,3).type_as(transparams).zero_()
                eMat[0][1] = -e_i[2]
                eMat[0][2] =  e_i[1]
                eMat[1][0] =  e_i[2]
                eMat[1][2] = -e_i[0]

                eMat[2][0] = -e_i[1]
                eMat[2][1] =  e_i[0]
                omega_skew[b] = eMat 
        return torch.bmm(omega_skew, RotMats)

        '''
		from http://arxiv.org/pdf/1312.0788.pdf

        v_i [v]x + [v x (Id - R)e_i]x 
	    ----------------------------  R 
            	 ||v||^{2}
        '''
    
    def forward(self,_tranformParams):

        self.save_for_backward(_tranformParams)

        self.transformParams = _tranformParams
        batchSize = self.transformParams.size(0) 
        completeTransformation = torch.zeros(batchSize,3,3).type_as(self.transformParams)
        completeTransformation.select(2,0).select(1,0).add_(1)
        completeTransformation.select(2,1).select(1,1).add_(1)
        completeTransformation.select(2,2).select(1,2).add_(1)

        # transformationBuffer = torch.Tensor(batchSize,3,3).type_as(self.transformParams)

        paramIndex = 0

        omega_x = self.transformParams.select(1,paramIndex)
        omega_y = self.transformParams.select(1,paramIndex + 1)
        omega_z = self.transformParams.select(1,paramIndex + 2)

        omega_skew = torch.Tensor(batchSize,3,3).type_as(self.transformParams)
        omega_skew.zero_()
        omega_skew.select(2,0).select(1,1).copy_(omega_z)	
        omega_skew.select(2,0).select(1,2).copy_(-omega_y)	

        omega_skew.select(2,1).select(1,0).copy_(-omega_z)	
        omega_skew.select(2,1).select(1,2).copy_(omega_x)
      
        omega_skew.select(2,2).select(1,0).copy_(omega_y)
        omega_skew.select(2,2).select(1,1).copy_(-omega_x)	
        #print(omega_skew)
        omega_skew_sqr = torch.bmm(omega_skew,omega_skew)
        #print(omega_skew_sqr)
        theta_sqr = torch.pow(omega_x,2) + torch.pow(omega_y,2) + torch.pow(omega_z,2)	
        theta = torch.pow(theta_sqr,0.5)	
        sin_theta = torch.sin(theta)
        sin_theta_div_theta = torch.div(sin_theta,theta)
        
        one_minus_cos_theta = torch.ones(theta.size()).type_as(self.transformParams) - torch.cos(theta)
        one_minus_cos_div_theta_sqr = torch.div(one_minus_cos_theta,theta_sqr)
        sin_theta_div_theta_tensor  = torch.ones(omega_skew.size()).type_as(self.transformParams)
        one_minus_cos_div_theta_sqr_tensor = torch.ones(omega_skew.size()).type_as(self.transformParams)
        for b in range(batchSize):
            if theta_sqr[b] > self.threshold:
                sin_theta_div_theta_tensor[b] = sin_theta_div_theta_tensor[b].fill_(sin_theta_div_theta[b])
                one_minus_cos_div_theta_sqr_tensor[b] = one_minus_cos_div_theta_sqr_tensor[b].fill_(one_minus_cos_div_theta_sqr[b])
            else:
                sin_theta_div_theta_tensor[b] = sin_theta_div_theta_tensor[b].fill_(1)
                one_minus_cos_div_theta_sqr_tensor[b] = one_minus_cos_div_theta_sqr_tensor[b].fill_(0)
        completeTransformation = completeTransformation + torch.mul(sin_theta_div_theta_tensor,omega_skew) + torch.mul(one_minus_cos_div_theta_sqr_tensor, omega_skew_sqr)
        

        self.rotationOutput = completeTransformation.narrow(1,0,3).narrow(2,0,3).clone()
        #print('self.rotationOutput')
        #print(self.rotationOutput)
        #print('completeTransformation')
        #print(completeTransformation)
        self.scaleOutput = completeTransformation.narrow(1,0,3).narrow(2,0,3).clone()

        self.output=completeTransformation.narrow(1,0,3)
        #print('dR_by_dv1..')
        #print(self.dR_by_dvi(self.transformParams,self.output,0, self.threshold))
        #print('dR_by_dv2..')
        #print(self.dR_by_dvi(self.transformParams,self.output,1, self.threshold))
        #print('dR_by_dv3..')
        #print(self.dR_by_dvi(self.transformParams,self.output,2, self.threshold))
        #print(self.output)
        return self.output
    
    def backward(self,_gradParams):
        #self.transformParams = _tranformParams
        #print(self.transformParams)
        gradParams = _gradParams
        batchSize = self.transformParams.size(0)
        
        paramIndex = self.transformParams.size(1)
        self.gradInput = torch.Tensor(self.transformParams.size()).zero_()
        
        rotationDerivative = torch.zeros(batchSize, 3, 3).type_as(self.transformParams)

        gradInputRotationParams = self.gradInput.narrow(1,0,1) #get the first number
        rotationDerivative = self.dR_by_dvi(self.transformParams,self.rotationOutput,0, self.threshold)	
        selectedGradParams = gradParams.narrow(1,0,3).narrow(2,0,3)
        gradInputRotationParams.copy_(torch.mul(rotationDerivative,selectedGradParams).sum())

        rotationDerivative = self.dR_by_dvi(self.transformParams,self.rotationOutput,1, self.threshold)	
        gradInputRotationParams = self.gradInput.narrow(1,1,1)
        gradInputRotationParams.copy_(torch.mul(rotationDerivative,selectedGradParams).sum())

        rotationDerivative = self.dR_by_dvi(self.transformParams,self.rotationOutput,2, self.threshold)
        gradInputRotationParams = self.gradInput.narrow(1,2,1)
        gradInputRotationParams.copy_(torch.mul(rotationDerivative,selectedGradParams).sum())

        #print('self.gradInput',self.gradInput)
        return self.gradInput
    













