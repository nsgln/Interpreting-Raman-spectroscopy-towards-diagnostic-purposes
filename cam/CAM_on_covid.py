import sys, os
from time import time

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from scripts.model_manager.model_script import *
from scripts.data_manager.data_script import *
from cam.CAM import *
from cam.importance import *

# ---------- Set up GPU environment ----------
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4";
gpus_list = [0, 1, 2, 3]

# ---------- Load data ----------
train_dataset, validation_dataset, test_dataset, _ = loadAndPrepareData("data/dataset_COVID_RAW.pkl", "train_settings/training_settings_cov_raw.pckl", False)

# ---------- Load pre-trained models ----------
numEpochs = 273
patience = 50
directoryToSaveModel = "saved_models/CAM_Covid"
models_list, device = trainingLoop(gpus_list, numEpochs, patience, train_dataset, validation_dataset, directoryToSaveModel, verbose=0, CAM=True)

# ---------- Compute CAM for each model ----------
globalDirectory = "CAMResults/COVID"
start = time()
for i in range(len(models_list)):
    directoryToSaveResults = globalDirectory + '/model' + str(i)
    model = models_list[i]
    dataset = test_dataset[i]
    loader = DataLoader(dataset, batch_size=1)
    finalConvLayer = model.module._modules.get('cnn_layers')[7]
    computeCAM(loader, directoryToSaveResults, finalConvLayer, model, gpus_list)
end = time()
hT = (end-start)//3600
mT = ((end-start)%3600)//60
sT = (((end-start)%3600)%60)
print("------------------------------ CAM computed for all models in {} h {} m and {} s ------------------------------".format(hT, mT, sT))