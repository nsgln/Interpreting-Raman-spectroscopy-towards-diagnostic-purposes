import os, sys
from time import time
from scipy.interpolate.fitpack import splint
from torch import flatten
from matplotlib.cbook import flatten as flat
import matplotlib.pyplot as plt
import numpy as np
from torch._C import dtype

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from cam.Grad_CAM import *
from torchcubicspline.interpolate import *
from cam.visualize_results import getHighestValues

""" Function to visualize the CAM result and save it in a file
    Param :
        - spectra : the raw spectra
        - cam : value calculated by cam
        - wellClassified : boolean, true if the model classify well this spectra, false otherwise
        - title : the title of the fig
        - pathToSave : absolute path, containing the name of the file, where the figure will be saved"""
def visualizeCAM(spectra, cam, wellClassified, title, pathToSave):
    plt.figure(figsize=(20,10))
    plt.subplot(311)
    plt.plot(spectra, 'k')
    plt.legend(['Raw spectra'])
    plt.xlabel('Raman shift (cm-1)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Original spectra')
    plt.subplot(312)
    if wellClassified:
        plt.plot(cam, 'g')
    else:
        plt.plot(cam, 'r')
    plt.legend(['CAM'])
    plt.xlabel('Raman shift (cm-1)')
    plt.ylabel('Importance (a.u.)')
    plt.title('Gradient Class Activation Mapping of the spectra')
    plt.subplot(313)
    valueToHighlight = getHighestValues(cam, 10)
    x_value = [x for x, y in valueToHighlight]
    if wellClassified:
        color = 'palegreen'
    else:
        color = 'lightcoral'
    plt.plot(spectra, 'k')
    for i in range(len(x_value)):
        plt.axvspan(x_value[i]-0.2, x_value[i]+0.2, color=color, label="_"*i+"Most important variables")
    plt.legend(['Raw spectra'])
    plt.xlabel('Raman shift (cm-1)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Original spectra with highlight on most important variable')
    plt.suptitle(title)
    plt.savefig(pathToSave)
    plt.close('all')

""" Function to save the results obtained.
    Params :
        - cam : the cam value calculated. 
        - pathToSave : the path of the file. """
def saveResults(cam, pathToSave):
    file = open(pathToSave, 'w')
    for elt in cam:
        file.write(str(elt[0]) + " ")
    file.close()

""" Function to compute Grad-CAM on an entire dataset
    Params :
        - testLoader : the dataloader of dataset on which CAM is computed.
        - directoryToSaveResults : the directory where the results are savec.
        - convolutionalPart : the convolutionalPart of the pretrained model including the last convolutional layer.
        - taskPart : the layers after the last convolutional layer, which are dedicated to the task.
        - device : the device on which the model is trained.
        - pool : an eventual pool layer, there is often one just before the last convolutional layer. 
        - model : the name of the model between covid and bacteria. """
def computeGrad_CAM(testLoader, directoryToSaveResults, convolutionalPart, taskPart, device, pool=None, model='covid'):
    if not (os.path.exists(directoryToSaveResults)):
        os.makedirs(directoryToSaveResults)
    
    globalTime = time()

    gCAM = Grad_CAM(convolutionalPart, taskPart, pool, model)

    gCAM.to(device)
    gCAM.eval()
    
    for idx, (spectra, label) in enumerate(testLoader):
        directoryToSaveResults2 = directoryToSaveResults + '/' + str(int(label))
        if not(os.path.exists(directoryToSaveResults2)):
            os.mkdir(directoryToSaveResults2)

        directoryForGraphs = directoryToSaveResults2 + '/Graphs'
        if not(os.path.exists(directoryForGraphs)):
            os.mkdir(directoryForGraphs)

        directoryForTxt = directoryToSaveResults2 + '/Values'
        if not(os.path.exists(directoryForTxt)):
            os.mkdir(directoryForTxt)

        spectra = spectra.to(device)
        prediction = gCAM(spectra)

        labelPredicted = prediction.argmax().cpu().numpy()

        nameOfGraph = 'Sprectra_' + str(idx) + '_predicted_as_' + str(labelPredicted) +'.png'
        pathOfGraph = directoryForGraphs + '/' + nameOfGraph

        nameOfTxt = 'Sprectra_' + str(idx) + '_predicted_as_' + str(labelPredicted) +'.txt'
        pathOfTxt = directoryForTxt + '/' + nameOfTxt

        if not(os.path.exists(pathOfGraph)) or not(os.path.exists(pathOfTxt)):
            inputSize = len(list(flatten(spectra).cpu().numpy()))

            prediction[:,labelPredicted].backward()

            gradients = gCAM.getActivationsGradients()
            pooled_gradients = torch.mean(gradients, dim=2)

            feature_maps = gCAM.getActivations(spectra).detach()

            for k in range(list(feature_maps.size())[1]):
                feature_maps[:, k, :] = torch.mul(feature_maps[:, k, :], pooled_gradients[:, k].cpu().numpy()[0])
        
            sizeFeatureMap = len(flatten(feature_maps.squeeze()).cpu().numpy())
            featureMapFlat = flatten(feature_maps.squeeze()).cpu()
            featureMapFlat = torch.unsqueeze(featureMapFlat, 1)
            x = torch.linspace(0, sizeFeatureMap-1, sizeFeatureMap)
            coeffSpline = natural_cubic_spline_coeffs(x, featureMapFlat)
            spline = NaturalCubicSpline(coeffSpline)
            x_wanted = torch.linspace(0, sizeFeatureMap-1, inputSize)
            vector = spline.evaluate(x_wanted)

            if (model == 'covid'):
                grad_cam = torch.where(vector > 0, vector, 0.)
            else:
                grad_cam = torch.where(vector > 0, vector, torch.tensor(0., dtype=torch.float32))
            grad_cam = grad_cam.numpy()

            #Save results
            wellClassified = (int(label.cpu()) == labelPredicted)
            title = 'Sprectra ' + str(idx) + ' predicted as ' + str(prediction[0])
            spectra = spectra.cpu().numpy()
            spectra = list(flat(spectra))
            visualizeCAM(spectra, grad_cam, wellClassified, title, pathOfGraph)
            saveResults(grad_cam, pathOfTxt)
    
    end = time()
    hT = (end-globalTime)//3600
    mT = ((end-globalTime)%3600)//60
    sT = (((end-globalTime)%3600)%60)
    print("Done in {} h {} m and {} s.".format(hT, mT, sT))