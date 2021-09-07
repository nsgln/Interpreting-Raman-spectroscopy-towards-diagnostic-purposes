import sys, os
from time import time
import torch
import numpy as np
from scipy import interpolate

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from scripts.model_manager.model_script import *
from scripts.data_manager.data_script import *
from cam.compute_Grad_CAM import *

# ---------- Set up GPU environment ----------
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4";
gpus_list = [0, 1, 2, 3]

# ---------- Load data ----------
train_dataset, validation_dataset, test_dataset, _ = loadAndPrepareData("data/dataset_COVID_RAW.pkl", "train_settings/training_settings_cov_raw.pckl", False)

# ---------- Load pre-trained models ----------
numEpochs = 273
patience = 50
directoryToSaveModel = "saved_models/covid_RAW"
models_list, device = trainingLoop(gpus_list, numEpochs, patience, train_dataset, validation_dataset, directoryToSaveModel, verbose=0)

# ---------- Compute grad-CAM for each model ----------
globalDirectory = "Grad-CAMResults/COVID"
start = time()
for i in range(len(models_list)):
    directoryToSaveResults = globalDirectory + '/model' + str(i)
    model = models_list[i]
    dev = device[i]
    dataset = test_dataset[i]
    loader = DataLoader(dataset, batch_size=1)
    convolutionalPart = model.module._modules.get('cnn_layers')[:9]
    taskPart = model.module._modules.get('dense_layers')
    pool = model.module._modules.get('cnn_layers')[9]
    computeGrad_CAM(loader, directoryToSaveResults, convolutionalPart, taskPart, dev, pool)
end = time()
hT = (end-start)//3600
mT = ((end-start)%3600)//60
sT = (((end-start)%3600)%60)
print("------------------------------ CAM computed for all models in {} h {} m and {} s ------------------------------".format(hT, mT, sT))

""" # ---------- Load one model for test ----------
pathToModel = "saved_models/covid_RAW/1.pckl"
model, device = loadModel(pathToModel, gpus_list)

# ---------- Load data ----------
train_dataset, validation_dataset, test_dataset, _ = loadAndPrepareData("data/dataset_COVID_RAW.pkl", "train_settings/training_settings_cov_raw.pckl", False)

# ---------- Test Grad-CAM
convolutionalPart = model.module._modules.get('cnn_layers')[:9]
#print("CONV PART :", convolutionalPart)
taskPart = model.module._modules.get('dense_layers')
#print("TASK PART :", taskPart)
pool = model.module._modules.get('cnn_layers')[9]
#print("POOL :", pool)
gradCAM = Grad_CAM(convolutionalPart, taskPart, pool)
gradCAM.to(device)

gradCAM.eval()

dataset = test_dataset[0]
testLoader = DataLoader(dataset, batch_size=1)
spectra, label = next(iter(testLoader))

spectra = spectra.to(device)

#prediction = gradCAM(spectra).argmax()
prediction = gradCAM(spectra)
# 2 = prediction.argmax() !
#print(prediction[:,2])
prediction[:,2].backward()

gradients = gradCAM.getActivationsGradients()
#print(gradients.size())
#print(gradients.size())

pool_gradients = torch.mean(gradients, dim=2)
#print(pool_gradients.size())

activations = gradCAM.getActivations(spectra).detach()
#print(activations.size())

for i in range(25):
    activations[:, i, :] = torch.mul(activations[:, i, :], pool_gradients[:,i].cpu().numpy()[0])

print(activations.size())

actlist = torch.flatten(activations.squeeze()).cpu().numpy()
plt.plot(actlist)
plt.savefig("activations.jpg")
plt.close()

x = [i for i in range(len(actlist))]
tck = interpolate.splrep(x, actlist, s=0, k=2)

xnew = np.linspace(0, len(x), 991)
ynew = interpolate.splev(xnew, tck)

plt.plot(ynew)
plt.savefig("interpolation.jpg")
plt.close()

result = torch.where(activations > 0, activations, 0.)
result = result.squeeze()
result = torch.flatten(result)

plt.plot(result.cpu().numpy())
plt.savefig("relu_activations.jpg")
plt.close()

#LGradCam = np.maximum(activations.cpu().numpy(), 0)
#print(LGradCam.size())
 """