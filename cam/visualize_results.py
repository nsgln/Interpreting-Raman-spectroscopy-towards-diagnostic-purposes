import collections
import operator
import os
from collections import Counter
from posix import WIFCONTINUED
import matplotlib.pyplot as plt

""" Function to read and parse the "Values" files created by the CAM and Grad-CAM methods.
    Params :
        - pathOfFile : the path of the value file."""
def parseResults(pathOfFile):
    f = open(pathOfFile, 'r')
    lines = f.readlines()
    f.close()
    value = lines[-1]
    values_str = value.split()
    values = [float(elt) for elt in values_str]
    return values

""" Function to get the x percents of highest values in a list.
    Params :
        - values : the list of values.
        - percents : the percents wanted - int (10 = 10% of highest values). """ 
def getHighestValues(values, percents):
    m = len(values)
    p = percents / 100
    n = int(m * p)
    if n == 0:
        n = 1
    
    indexed = list(enumerate(values))
    top = sorted(indexed, key=operator.itemgetter(1))
    topn = top[-n:]
    return list(reversed(topn))

def getHighestInFile(pathOfFile):
    values = parseResults(pathOfFile)
    topValues = getHighestValues(values, 10)
    x_values = [x for x,y in topValues]
    return x_values

def getHighest(pathOfDir, numClasses):
    numOfFile = dict()
    result = dict()
    for i in range(numClasses):
        result[i] = []
        numOfFile[i] = 0
    listOfElements = os.listdir(pathOfDir)
    for elt in listOfElements:
        path = pathOfDir + '/' + elt
        if os.path.isfile(path):
            st = path.find('s_') + 2
            end = path.find('.')
            c = int(path[st : end])
            numOfFile[c] += 1
            result[c].extend(getHighestInFile(path))
        elif os.path.isdir(path):
            if elt != "Graphs" and elt != "Statistics":
                resultOfSubDir, numOfFileSubDir = getHighest(path, numClasses)
                keysOfSubDir = resultOfSubDir.keys()
                for k in keysOfSubDir:
                    result[k] = result[k] + resultOfSubDir[k]
                    numOfFile[k] += numOfFileSubDir[k]
    return result, numOfFile

def getHighestByWindow(pathOfDir, numClasses, windowSize, inputSizeSpectra):
    numOfFile = dict()
    result = dict()
    for i in range(numClasses):
        result[i] = dict()
        numOfFile[i] = 0
    listOfElements = os.listdir(pathOfDir)
    for elt in listOfElements:
        path = pathOfDir + '/' + elt
        if os.path.isfile(path):
            st = path.find('s_') + 2
            end = path.find('.')
            c = int(path[st : end])
            numOfFile[c] += 1
            tmp = sorted(getHighestInFile(path))
            count = 0
            v = windowSize // 2
            while count < inputSizeSpectra:
                test = [count+i for i in range(windowSize)]
                if any(elt in test for elt in tmp):
                    if v in result[c].keys():
                        result[c][v] = result[c][v] + 1
                    else:    
                        result[c][v] = 1
                count = count + windowSize
                v = v + windowSize
        elif os.path.isdir(path):
            if elt != "Graphs" and not elt.startswith("Statistics"):
                resultOfSubDir, numOfFileSubDir = getHighestByWindow(path, numClasses, windowSize, inputSizeSpectra)
                keysOfSubDir = resultOfSubDir.keys()
                for k in keysOfSubDir:
                    keysOfSubDir2 = resultOfSubDir[k].keys()
                    for k2 in keysOfSubDir2:
                        if k2 in result[k].keys():
                            result[k][k2] = result[k][k2] + resultOfSubDir[k][k2]
                        else:
                            result[k][k2] = resultOfSubDir[k][k2]
                    numOfFile[k] += numOfFileSubDir[k]
    return result, numOfFile

def statistics(pathOfDir, numClasses):
    result, numOfFile = getHighest(pathOfDir, numClasses)
    keyOfResults = result.keys()
    for k in keyOfResults:
        lst = result[k]
        numSpectra = numOfFile[k]
        c = Counter(lst)
        pathStat = pathOfDir + '/Statistics'
        pathTxt = pathStat + '/Value'
        if not(os.path.exists(pathTxt)):
            os.makedirs(pathTxt)
        path = pathTxt + '/statisticsForClass' + str(k) + '.txt'
        pathGraph = pathStat + '/BarChart'
        if not(os.path.exists(pathGraph)):
            os.makedirs(pathGraph)
        graphPath = pathGraph + '/barChartForClass' + str(k) + '.png'
        graphPath2 = pathGraph + '/barChartForClassMajor' + str(k) + '.png'
        if not(os.path.exists(graphPath)) or not(os.path.exists(path)) or not(os.path.exists(graphPath2)):
            x = list(c.keys())
            values = list(c.values())
            for i in range(len(values)):
                values[i] = (values[i]/numSpectra) * 100

            colors = ['cornflowerblue' if (x < 70) else 'lightcoral' for x in values]
            plt.figure(figsize=(20,10))
            plt.bar(x, values, width=1.2, color=colors)
            plt.xlabel('Value of Raman shift (cm-1)')
            plt.ylabel('Importance in classification (%)')
            plt.title('Percentage of spectra where a variable is used for classification as class ' + str(k))
            plt.savefig(graphPath)
            plt.close('all')

            variableP = [x[i] for i in range(len(values)) if (values[i] >= 70)]
            valuesP = [values[i] for i in range(len(values)) if (values[i] >= 70)]
            variableM = [x[i] for i in range(len(values)) if (values[i] < 70)]
            file = open(path, 'w')
            file.write(str(c))
            file.write('\n')
            file.write('Variable >= 70% : ' + str(variableP))
            file.write('\n')
            file.write('Variable < 70% : ' + str(variableM))
            file.close()

            plt.figure(figsize=(20,10))
            plt.bar(variableP, valuesP, width=1.2, color='salmon')
            plt.xlabel('Value of Raman shift (cm-1)')
            plt.ylabel('Importance in classification (%)')
            plt.title('Most used variable for classification as class ' + str(k))
            plt.savefig(graphPath2)
            plt.close('all')

def functionStatisticsForModel(pathOfDir, numClasses):
    result, numOfFile = getHighest(pathOfDir, numClasses)
    keyOfResults = result.keys()
    for k in keyOfResults:
        lst = result[k]
        numSpectra = numOfFile[k]
        c = Counter(lst)
        pathStat = pathOfDir + '/Statistics'
        pathTxt = pathStat + '/Value'
        if not(os.path.exists(pathTxt)):
            os.makedirs(pathTxt)
        path = pathTxt + '/statisticsForClass' + str(k) + '.txt'
        pathGraph = pathStat + '/BarChart'
        if not(os.path.exists(pathGraph)):
            os.makedirs(pathGraph)
        graphPath = pathGraph + '/barChartForClass' + str(k) + '.png'
        graphPath2 = pathGraph + '/barChartForClassMajor' + str(k) + '.png'
        if not(os.path.exists(graphPath)) or not(os.path.exists(path)) or not(os.path.exists(graphPath2)):
            x = list(c.keys())
            values = list(c.values())
            for i in range(len(values)):
                values[i] = (values[i]/numSpectra) * 100

            colors = ['cornflowerblue' if (x < 70) else 'lightcoral' for x in values]
            plt.figure(figsize=(20,10))
            plt.bar(x, values, width=1.2, color=colors)
            plt.xlabel('Value of Raman shift (cm-1)')
            plt.ylabel('Importance in classification (%)')
            plt.title('Percentage of spectra where a variable is used for classification as class ' + str(k))
            plt.savefig(graphPath)
            plt.close('all')

            variableP = [x[i] for i in range(len(values)) if (values[i] >= 70)]
            valuesP = [values[i] for i in range(len(values)) if (values[i] >= 70)]
            variableM = [x[i] for i in range(len(values)) if (values[i] < 70)]
            file = open(path, 'w')
            file.write(str(c))
            file.write('\n')
            file.write('Variable >= 70% : ' + str(variableP))
            file.write('\n')
            file.write('Variable < 70% : ' + str(variableM))
            file.close()

def StatisticsForEachModel(pathOfDir, numClasses):
    for i in range(101):
        pathOfModel = pathOfDir + '/model' + str(i)
        functionStatisticsForModel(pathOfModel, numClasses)

def statisticsByWindows(pathOfDir, numClasses, windowSize, inputSize):
    result, numOfFile = getHighestByWindow(pathOfDir, numClasses, windowSize, inputSize)
    keyOfResults = result.keys()
    for k in keyOfResults:
        lst = result[k]
        numSpectra = numOfFile[k]
        od = collections.OrderedDict(sorted(lst.items()))
        pathStat = pathOfDir + '/StatisticsByWindowsOfSize' + str(windowSize)
        pathTxt = pathStat + '/ValueByWindows'
        if not(os.path.exists(pathTxt)):
            os.makedirs(pathTxt)
        path = pathTxt + '/statisticsForClass' + str(k) + '.txt'
        pathGraph = pathStat + '/BarChart'
        if not(os.path.exists(pathGraph)):
            os.makedirs(pathGraph)
        graphPath = pathGraph + '/barChartForClass' + str(k) + '.png'
        if not(os.path.exists(graphPath)) or not(os.path.exists(path)):
            x = list(od.keys())
            values = list(od.values())
            for i in range(len(values)):
                values[i] = (values[i]/numSpectra) * 100

            colors = ['cornflowerblue' if (x < 70) else 'lightcoral' for x in values]
            plt.figure(figsize=(20,10))
            plt.bar(x, values, width=windowSize, color=colors, edgecolor = "black")
            plt.xlabel('Variable of Raman shift (cm-1)')
            plt.ylabel('Importance in classification (%)')
            plt.title('Percentage of spectra where a region is used for classification as class ' + str(k))
            plt.savefig(graphPath)
            plt.close('all')

            variableP = [x[i] for i in range(len(values)) if (values[i] >= 70)]
            variableM = [x[i] for i in range(len(values)) if (values[i] < 70)]
            file = open(path, 'w')
            file.write(str(od))
            file.write('\n')
            file.write('Variable >= 70% : ' + str(variableP))
            file.write('\n')
            file.write('Variable < 70% : ' + str(variableM))
            file.close()