from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
import os, sys
from time import time
import torch
import numpy as np

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from cam.importance import *
from cam.visualize_results import getHighestValues

""" Function to visualize the CAM result and save it in a file
    Param :
        - spectra : the raw spectra
        - cam : value calculated by cam
        - wellClassified : boolean, true if the model classify well this spectra, false otherwise
        - title : the title of the fig
        - pathToSave : absolute path, containing the name of the file, where the figure will be saved"""
def visualizeCAM(spectra, cam, wellClassified, title, pathToSave):
    minmaxCAM = minmax_scale(cam)
    plt.figure(figsize=(20,20))
    plt.subplot(311)
    plt.plot(spectra, 'k')
    plt.legend(['Raw spectra'])
    plt.xlabel('Raman shift (cm-1)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Original spectra')
    plt.subplot(312)
    if wellClassified:
        plt.plot(minmaxCAM, 'g')
    else:
        plt.plot(minmaxCAM, 'r')
    plt.legend(['CAM'])
    plt.xlabel('Raman shift (cm-1)')
    plt.ylabel('Importance (a.u.)')
    plt.title('Class Activation Mapping of the spectra')
    plt.subplot(313)
    valueToHighlight = getHighestValues(minmaxCAM, 10)
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
    return minmaxCAM

""" Function to save the results obtained.
    Params :
        - cam : the cam value calculated. 
        - minmaxcam : the 'minmaxed' cam value.
        - pathToSave : the path of the file. """
def saveResults(cam, minmaxcam, pathToSave):
    file = open(pathToSave, 'w')
    for elt in cam:
        file.write(str(elt.numpy()[0]) + " ")
    file.write('\n')
    for minmax in minmaxcam:
        file.write(str(minmax) + " ")
    file.close()

""" Function to get the features map
    Param : 
        - finalConvLayer : the last convolutional layer to extract features maps """
def getFeaturesMaps(finalConvLayer):
    features_maps = []
    #Function to get features map
    def hook_feature(module, input, output):
        features_maps.append(output.cpu().data.numpy())
    #Add this function to the last conv layer
    finalConvLayer.register_forward_hook(hook_feature)
    return features_maps

""" Function to compute CAM on a entire dataset
    Params : 
        - testLoader : the dataLoader of test set.
        - directoryToSaveResults : the directory where results are saved.
        - finalConvLayer : the last convolutional layer of the model.
        - model : the pretrained model.
        - gpus_list : the list of the id of gpu used. """
def computeCAM(testLoader, directoryToSaveResults, finalConvLayer, model, gpus_list):
    if not (os.path.exists(directoryToSaveResults)):
        os.makedirs(directoryToSaveResults)
    globalTime = time()
    for idx, (ramanSpectra, label) in enumerate(testLoader):
        directoryToSaveResults2 = directoryToSaveResults + '/' + str(int(label))
        if not(os.path.exists(directoryToSaveResults2)):
            os.mkdir(directoryToSaveResults2)

        directoryForGraphs = directoryToSaveResults2 + '/Graphs'
        if not(os.path.exists(directoryForGraphs)):
            os.mkdir(directoryForGraphs)

        directoryForTxt = directoryToSaveResults2 + '/Values'
        if not(os.path.exists(directoryForTxt)):
            os.mkdir(directoryForTxt)
        
        inputSize = len(list(flatten(ramanSpectra.cpu().numpy())))

        # Get the feature maps
        featuresMaps = getFeaturesMaps(finalConvLayer)

        # Use gpu
        cuda = torch.cuda.is_available()
        if cuda:
            gpu = 'cuda:' + str(gpus_list[0])
        device = torch.device(gpu if cuda else 'cpu')

        # Make prediction
        ramanSpectra = ramanSpectra.to(device)
        label = label.to(device)

        #model = model.double()
        model = model.to(device)
        model.eval()
        output = model(ramanSpectra)
        _, prediction = torch.max(output.data, 1)

        if cuda:
            prediction = prediction.cpu()
        prediction = prediction.numpy()

        nameOfGraph = 'Sprectra_' + str(idx) + '_predicted_as_' + str(prediction[0]) +'.png'
        pathOfGraph = directoryForGraphs + '/' + nameOfGraph

        nameOfTxt = 'Sprectra_' + str(idx) + '_predicted_as_' + str(prediction[0]) +'.txt'
        pathOfTxt = directoryForTxt + '/' + nameOfTxt

        if not(os.path.exists(pathOfGraph)) or not(os.path.exists(pathOfTxt)):
            # Get weights
            named_params = list(model.named_parameters())
            weights = np.squeeze(named_params[-2][1].data.cpu().numpy())

            # Reshape features maps
            #print(featuresMaps)
            x, y, a, b = np.shape(featuresMaps)
            #x, y, a = np.shape(featuresMaps)

            features_maps = np.reshape(featuresMaps, (x*y, a*b))
            #features_maps = np.reshape(featuresMaps, (x*y, a))
            a, b = np.shape(features_maps)
            features_maps = np.reshape(features_maps, (a, b, 1))
            features_maps = torch.tensor(features_maps)

            # Compute importance
            imp = Importance(features_maps, weights, inputSize, prediction[0])
            imp = imp.to(device)
            imp.computeVectors()
            """plt.plot(imp.vectors[0])
            plt.savefig(pathOfGraph)
            plt.close('all')"""

            # Compute CAM
            cam = []
            for x, y in enumerate(list(flatten(ramanSpectra.cpu().numpy()))):
                cam.append(imp(x))
        
            # Save results
            wellClassified = (int(label.cpu()) == prediction[0])
            title = 'Sprectra ' + str(idx) + ' predicted as ' + str(prediction[0])
            spectra = ramanSpectra.cpu().numpy()
            spectra = list(flatten(spectra))
            minmaxCam = visualizeCAM(spectra, cam, wellClassified, title, pathOfGraph)
            saveResults(cam, minmaxCam, pathOfTxt)
    
    end = time()
    hT = (end-globalTime)//3600
    mT = ((end-globalTime)%3600)//60
    sT = (((end-globalTime)%3600)%60)
    print("Done in {} h {} m and {} s.".format(hT, mT, sT))