import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

""" Class to compute Gradient CAM method"""
class Grad_CAM(nn.Module):
    """ Init of the class
        Param :  
            - convolutionalPart : the convolutionalPart of the pretrained model including the last convolutional layer
            - taskPart : the layers after the last convolutional layer, which are dedicated to the task
            - pool : an eventual pool layer, there is often one just before the last convolutional layer
            - model : the name of the model between covid and bacteria """
    def __init__(self, convolutionalPart, taskPart, pool = None, model='covid'):
        super(Grad_CAM, self).__init__()
        self.conv = convolutionalPart
        self.task = taskPart
        self.pool = pool
        self.gradients = None
        self.model = model

    """ Getter for the gradients """
    def getActivationsGradients(self):
        return self.gradients

    """ Method to extract the activation
        Param :
            - x : the data """
    def getActivations(self, x):
        return self.conv(x)

    """ Hook for the gradients of the activations """
    def gradient_activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        if(self.model == 'covid'):
            # apply the convolutional layers
            x = x.resize_(x.shape[0], 1, x.shape[1])
            x = self.conv(x)

            # register the hook
            hook = x.register_hook(self.gradient_activations_hook)

            # apply the remaining part of the model
            if self.pool != None :
                x = self.pool(x)
        
            x = torch.flatten(x, 1)
            x = self.task(x)

        else:
            x = F.relu(self.conv(x))

            hook = x.register_hook(self.gradient_activations_hook)

            if self.pool != None :
                x = self.pool(x)
            
            x = x.view(x.size(0), -1)

            x = self.task(x)
        
        return x