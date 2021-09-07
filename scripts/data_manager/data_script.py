import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir2)

from scripts.data_manager.RamanDataset import RamanDataset
from scripts.data_manager.Datasets import Datasets
import pandas as pd
from os import path
import numpy as np
import copy
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
import pickle

def loadData(pathToData, withExternalGroup):
	df, labels, names = pd.read_pickle(pathToData)
	df = pd.DataFrame(df)
	labels = pd.Series(labels, name='label')
	names = pd.Series(names, name='names')
	df = pd.concat((df, labels, names), axis=1)

	if withExternalGroup:
		outside_group = np.random.choice(df.names.values)
		df = df[df.names.values != outside_group]
		final_test_set = df[df.names.values == outside_group]
		df = df[df.names.values != outside_group]
		X_set = df.drop(columns=['label', "names"]).values
		Y_set = df.label.values
		groups = df.names.values
		folds = list(LeaveOneGroupOut().split(X_set, Y_set, groups=groups))

	else:
		X_set = df.drop(columns=['label', "names"]).values
		Y_set = df.label.values
		groups = df.names.values
		folds = list(LeaveOneGroupOut().split(X_set, Y_set, groups=groups))
		final_test_set = None

	return X_set, Y_set, groups, folds, final_test_set

def dataAugment(signal, betashift = 0.24039033704204857, slopeshift = 0.5640435054299953, multishift = 0.0013960388613510225):
	#baseline shift
	beta = np.random.random(size=(signal.shape[0],1))*2*betashift-betashift
	slope = np.random.random(size=(signal.shape[0],1))*2*slopeshift-slopeshift + 1
	#relative positions
	axis = np.array(range(signal.shape[1]))/float(signal.shape[1])
	#offset
	offset = slope*(axis) + beta - axis - slope/2. + 0.5

	#multiplicative coefficient
	multi = np.random.random(size=(signal.shape[0],1))*2*multishift-multishift + 1
	augmented_signal = multi*signal + offset

	return augmented_signal

def setsCreation(pathToFile, X_set, Y_set, folds):
	if not path.exists(pathToFile):
		train_set = []
		validation_set = []
		test_set = []

		for i, (train_idx, test_idx) in enumerate(folds):
			X_train_tmp = X_set[train_idx]
			Y_train_tmp = Y_set[train_idx]

			X_test_tmp = X_set[test_idx]
			Y_test_tmp = Y_set[test_idx]

			X_train_tmp, X_val_tmp, Y_train_tmp, Y_val_tmp = train_test_split(X_train_tmp, Y_train_tmp, test_size=.1,
			                                                                  stratify=Y_train_tmp)
			augment = 30
			augmented_data = []
			Y_list = copy.copy(Y_train_tmp)
			for i in range(augment):
				augmented_data.append(dataAugment(X_train_tmp))
			for i in range(augment - 1):
				Y_list = np.concatenate((Y_list, Y_train_tmp), axis=0)

			X_train_tmp = np.vstack(augmented_data)
			Y_train_tmp = copy.copy(Y_list)
			train_set_tmp = RamanDataset(X_train_tmp, Y_train_tmp)
			train_set.append(train_set_tmp)
			val_set_tmp = RamanDataset(X_val_tmp, Y_val_tmp)
			validation_set.append(val_set_tmp)
			test_set_tmp = RamanDataset(X_test_tmp, Y_test_tmp)
			test_set.append(test_set_tmp)

		train_dataset = Datasets(train_set)
		validation_dataset = Datasets(validation_set)
		test_dataset = Datasets(test_set)
		training_settings = (train_dataset, validation_dataset, test_dataset)

		with open(pathToFile, "wb") as outf:
			pickle.dump(training_settings, outf)

	else:
		with open(pathToFile, "rb") as inf:
			train_dataset, validation_dataset, test_dataset = pickle.load(inf)

	return train_dataset, validation_dataset, test_dataset

def loadAndPrepareData(pathToData, pathToTrainingFile, withExternalGroup):
	X_set, Y_set, groups, folds, final_test_set = loadData(pathToData, withExternalGroup)
	if final_test_set != None :
		final_X_set = final_test_set.drop(columns=['label', "names"]).values
		final_Y_set = final_test_set.label.values
		final_test_dataset = RamanDataset(final_X_set, final_Y_set)
	else:
		final_test_dataset = None
	train_dataset, validation_dataset, test_dataset = setsCreation(pathToTrainingFile, X_set, Y_set, folds)

	return train_dataset, validation_dataset, test_dataset, final_test_dataset