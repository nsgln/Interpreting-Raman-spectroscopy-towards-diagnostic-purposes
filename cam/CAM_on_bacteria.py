import os, sys
import numpy as np

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from bacteria.resnetCAM import ResNetCAM
from bacteria.datasets import spectral_dataloader
from torchcubicspline.interpolate import *
from cam.importance import *
from cam.CAM import *

# ---------- Set up GPU environment ----------
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4";
gpus_list = [0, 1, 2, 3]

# ---------- Load Pre-trained model ----------
# CNN parameters
layers = 6
hidden_size = 100
block_size = 2
hidden_sizes = [hidden_size] * layers
num_blocks = [block_size] * layers
input_dim = 1000
in_channels = 64
n_classes = 30
cuda = torch.cuda.is_available()

cnn = ResNetCAM(hidden_sizes, num_blocks, input_dim,
            in_channels=in_channels, n_classes=n_classes)

if cuda: cnn.cuda()
cnn.load_state_dict(torch.load(
    'bacteria/CAMModelFine-Tuned.ckpt'))

# ---------- Load data ----------
X_fn = 'bacteria/data_test/X_test.npy'
y_fn = 'bacteria/data_test/y_test.npy'
X = np.load(X_fn)
Y = np.load(y_fn)

# ---------- Run CAM ----------
testLoader = spectral_dataloader(X, Y, batch_size=1)
directoryToSaveResult = "CAMResults/Bacteria"
finalConvLayer = cnn._modules.get('encoder')[-1][-1]._modules.get('conv2')
computeCAM(testLoader, directoryToSaveResult, finalConvLayer, cnn, gpus_list)