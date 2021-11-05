import sys, os
from time import time

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from cam.visualize_results import statisticsByWindows

start = time()
statisticsByWindows("CAMResults/COVID", 3, 50, 991)
end = time()
hT = (end-start)//3600
mT = ((end-start)%3600)//6
sT = (((end-start)%3600)%60)
print("------------------------------ Statistics of CAM on Covid dataset computed in in {} h {} m and {} s ------------------------------".format(hT, mT, sT))
start = time()
statisticsByWindows("Grad-CAMResults/COVID", 3, 50, 991)
end = time()
hT = (end-start)//3600
mT = ((end-start)%3600)//60
sT = (((end-start)%3600)%60)
print("------------------------------ Statistics of Grad-CAM on Covid dataset computed in in {} h {} m and {} s ------------------------------".format(hT, mT, sT))