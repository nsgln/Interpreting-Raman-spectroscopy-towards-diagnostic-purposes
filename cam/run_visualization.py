import sys, os
from time import time

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from cam.visualize_results import StatisticsForEachModel, statistics

start = time()
statistics("CAMResults/Bacteria", 30)
end = time()
hT = (end-start)//3600
mT = ((end-start)%3600)//60
sT = (((end-start)%3600)%60)
print("------------------------------ Statistics of CAM on Bacteria dataset computed in in {} h {} m and {} s ------------------------------".format(hT, mT, sT))
start = time()
statistics("Grad-CAMResults/Bacteria", 30)
end = time()
hT = (end-start)//3600
mT = ((end-start)%3600)//60
sT = (((end-start)%3600)%60)
print("------------------------------ Statistics of Grad-CAM on Bacteria dataset computed in in {} h {} m and {} s ------------------------------".format(hT, mT, sT))
start = time()
statistics("CAMResults/COVID", 3)
end = time()
hT = (end-start)//3600
mT = ((end-start)%3600)//60
sT = (((end-start)%3600)%60)
print("------------------------------ Statistics of CAM on Covid dataset computed in in {} h {} m and {} s ------------------------------".format(hT, mT, sT))
start = time()
statistics("Grad-CAMResults/COVID", 3)
end = time()
hT = (end-start)//3600
mT = ((end-start)%3600)//60
sT = (((end-start)%3600)%60)
print("------------------------------ Statistics of Grad-CAM on Covid dataset computed in in {} h {} m and {} s ------------------------------".format(hT, mT, sT))
start = time()
StatisticsForEachModel("CAMResults/COVID", 3)
end = time()
hT = (end-start)//3600
mT = ((end-start)%3600)//60
sT = (((end-start)%3600)%60)
print("------------------------------ Statistics of CAM on Covid dataset computed in in {} h {} m and {} s ------------------------------".format(hT, mT, sT))
start = time()
StatisticsForEachModel("Grad-CAMResults/COVID", 3)
end = time()
hT = (end-start)//3600
mT = ((end-start)%3600)//60
sT = (((end-start)%3600)%60)
print("------------------------------ Statistics of Grad-CAM on Covid dataset computed in in {} h {} m and {} s ------------------------------".format(hT, mT, sT))
