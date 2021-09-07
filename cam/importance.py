from torchcubicspline.interpolate import *
import torch.nn as nn
import torch
import numpy as np

""" Class to compute the Class Activation Map for a spectra using Pytorch """
class Importance(nn.Module):
    """ Init method.
        Params :
            - featuresMaps : the features maps extracted from the last convolutionnal layer of the model.
            - weights : the weights of the model, extracted from the last layer of the model.
            - inputSize : the size of the input spectra, i.e. the number of variable.
            - idClass : the predicted class. """
    def __init__(self, featuresMaps, weights, inputSize, idClass):
        super(Importance, self).__init__()
        self.featuresMaps = featuresMaps
        self.sizeFeaturesMaps = len(featuresMaps)
        self.weights = weights
        self.inputSize = inputSize
        #self.x_wanted = torch.linspace(0, inputSize-1, inputSize).int()
        self.vectors = torch.tensor(0)
        self.targetClass = idClass

    """ Forward method. 
        Params :
            - variable : the variable whose importance we want to know. """
    def forward(self, variable):
        imp = 0
        for i in range(self.sizeFeaturesMaps):
            imp += self.vectors[i][variable] * self.weights[self.targetClass][i]
        return imp
    
    """ Method in order to compute the spline interpolation of each features maps. """
    def computeVectors(self):
        vectors = []
        for i in range(self.sizeFeaturesMaps):
            values = torch.linspace(0, len(self.featuresMaps[i])-1, len(self.featuresMaps[i]))
            coeffsSpline = natural_cubic_spline_coeffs(values, self.featuresMaps[i])
            spline = NaturalCubicSpline(coeffsSpline)
            x_wanted = torch.linspace(0, len(self.featuresMaps[i])-1, self.inputSize)
            vector = spline.evaluate(x_wanted)
            vectors.append(vector.numpy())
        vectors = torch.tensor(vectors)
        self.vectors = vectors
