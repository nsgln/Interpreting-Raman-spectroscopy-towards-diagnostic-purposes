from decimal import Clamped
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir2)

from scripts.model_manager.ConvNet import ConvNet
from scripts.model_manager.ConvNetCAM import ConvNetCAM
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from seaborn import heatmap
from statistics import mean
import matplotlib.pyplot as plt
import time

def createCNN(gpu_ids, CAM=False):
	if CAM:
		model = ConvNetCAM()
	else:
		model = ConvNet()
	model = model.double()
	optimizer = Adam(model.parameters(), lr=0.00020441990333108206)
	loss = nn.CrossEntropyLoss()

	if torch.cuda.is_available():
		cuda = 'cuda:' + str(gpu_ids[0])
		model = nn.DataParallel(model, device_ids=gpu_ids)
		loss.cuda()
	device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
	model.to(device)
	return model, loss, optimizer, device

def visualizeTrain(modelName, pathToSave, train_loss, val_loss, train_acc, val_acc):
	plt.plot(train_acc, 'c-')
	plt.plot(train_loss, 'c--')
	plt.plot(val_acc, 'r-')
	plt.plot(val_loss, 'r--')
	plt.title(modelName)
	plt.ylabel('value')
	plt.xlabel('epoch')
	plt.legend(['trainning accuracy', 'training loss', 'validation accuracy', 'validation loss'], loc='upper left')
	plt.savefig(pathToSave+modelName)

def train(device, model, loss, optimizer, train_dataset, validation_dataset, epochs, patience, path, verbose=0,
          batch_size=338):
	train_losses = []
	val_losses = []
	train_acc = []
	val_acc = []
	min_val_loss = np.Inf
	max_val_acc = np.NINF
	epochs_no_improve_loss = 0
	epochs_no_improve_acc = 0
	if verbose == 1:
		verbScheduler = True
	else:
		verbScheduler = False
	scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=80, cooldown=10, verbose=verbScheduler)
	training_generator = DataLoader(train_dataset, batch_size=batch_size)
	validation_generator = DataLoader(validation_dataset, batch_size=batch_size)
	model.train()
	for epoch in range(epochs):
		loss_train_epoch = []
		acc_train_epoch = []
		for i, (ramanSpectraTrain, labelTrain) in enumerate(training_generator):
			ramanSpectraTrain = ramanSpectraTrain.to(device)
			labelTrain = labelTrain.to(device)

			optimizer.zero_grad()

			output_train = model(ramanSpectraTrain)

			loss_train = loss(output_train, labelTrain)
			loss_train_epoch.append(loss_train.cpu().item())

			loss_train.backward()
			optimizer.step()

			output_label = torch.argmax(output_train, dim=1)
			acc_train = accuracy_score(labelTrain.cpu().detach().numpy(), output_label.cpu().detach().numpy())
			acc_train_epoch.append(acc_train)

		loss_train = mean(loss_train_epoch)
		acc_train = mean(acc_train_epoch)
		train_losses.append(loss_train)
		train_acc.append(acc_train)

		with torch.no_grad():
			loss_val_epoch = []
			acc_val_epoch = []
			for j, (ramanSpectraVal, labelVal) in enumerate(validation_generator):
				ramanSpectraVal = ramanSpectraVal.to(device)
				labelVal = labelVal.to(device)

				output_val = model(ramanSpectraVal)

				loss_val = loss(output_val, labelVal)
				loss_val_epoch.append(loss_val.cpu().item())

				val_label = torch.argmax(output_val, dim=1)
				acc_val = accuracy_score(labelVal.cpu().detach().numpy(), val_label.cpu().detach().numpy())
				acc_val_epoch.append(acc_val)

			loss_val = mean(loss_val_epoch)
			acc_val = mean(acc_val_epoch)
		val_losses.append(loss_val)
		val_acc.append(acc_val)
		scheduler.step(loss_val)
		if acc_val > max_val_acc:
			epochs_no_improve_acc = 0
			max_val_acc = acc_val
			torch.save({'model_state_dict': model.state_dict(),
			            'optimizer_state_dict': optimizer.state_dict(),
			            'train_loss': train_losses,
			            'train_acc': train_acc,
			            'val_loss': val_losses,
			            'val_acc': val_acc}, path)
		else:
			epochs_no_improve_acc += 1

		if loss_val < min_val_loss:
			epochs_no_improve_loss = 0
			min_val_loss = loss_val
			torch.save({'model_state_dict': model.state_dict(),
			            'optimizer_state_dict': optimizer.state_dict(),
			            'train_loss': train_losses,
			            'train_acc': train_acc,
			            'val_loss': val_losses,
			            'val_acc': val_acc}, path)
		else:
			epochs_no_improve_loss += 1

		if verbose == 1:
			print(
				"Epoch {}:\t train loss : {}; train accuracy : {}; \n validation loss : {}; validation accuracy : {}".format(
					epoch + 1, loss_train, acc_train, loss_val, acc_val))

		if epochs_no_improve_loss >= patience and epochs_no_improve_acc >= patience:
			print("Early stopping at epoch {}".format(epoch + 1))
			break

	checkpoint = torch.load(path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	train_losses = checkpoint['train_loss']
	train_acc = checkpoint['train_acc']
	val_losses = checkpoint['val_loss']
	val_acc = checkpoint['val_acc']

	if verbose == 1:
		print("------------------------------ Final result of the model ! ------------------------------")
		print("Train loss : {}; Train accuracy : {}; \n Validation loss : {}; Validation accuracy : {}".format(
			train_losses[-1], train_acc[-1], val_losses[-1], val_acc[-1]))

	return train_losses, val_losses, train_acc, val_acc

def testModel(model, test_set, device, groups_test, patient_level_prediction, batch_size=1):
	test_acc = []
	test_generator = DataLoader(test_set, batch_size=batch_size)
	model.eval()
	for i, (ramanSpectra, label) in enumerate(test_generator):
		ramanSpectra = ramanSpectra.to(device)
		label = label.to(device)

		labelPredict = model(ramanSpectra)
		labelPredict = torch.argmax(labelPredict, dim=1)

		patient_level_prediction[groups_test[i]][labelPredict] += 1

		acc = accuracy_score(label.cpu().detach().numpy(), labelPredict.cpu().detach().numpy())
		test_acc.append(acc)

	return test_acc, patient_level_prediction

def globalTest(models, test_set, devices, batch_size=1):
	test_generator = DataLoader(test_set, batch_size=batch_size)
	predictions = [[] for _ in models]
	for i in range(len(models)):
		model = models[i]
		model.eval()
		for j, (ramanSpectra, label) in enumerate(test_generator):
			ramanSpectra = ramanSpectra.to(devices[i])

			labelPredict = model(ramanSpectra)
			labelPredict = torch.argmax(labelPredict, dim=1)
			predictions[i].append(labelPredict)

	finalPrediction = []
	for i in range(len(predictions[0])):
		count0 = 0
		count1 = 0
		count2 = 0
		for j in range(len(predictions)):
			if predictions[j][i] == 0:
				count0 += 1
			elif predictions[j][i] == 1:
				count1 += 1
			else:
				count2 += 1
		m = max(count0, count1, count2)
		if m == count0:
			finalPrediction.append(0)
		elif m == count1:
			finalPrediction.append(1)
		else:
			finalPrediction.append(2)

	realLabels = []
	for j, (ramanSpectra, label) in enumerate(test_generator):
		realLabels.append(label)

	num0 = finalPrediction.count(0)
	num1 = finalPrediction.count(1)
	num2 = finalPrediction.count(2)
	m = max(num0, num1, num2)
	pred = ""
	if m == num0:
		pred = "Cov +"
	elif m == num1:
		pred = "Cov-"
	else:
		pred = "Control"
	acc = accuracy_score(realLabels, finalPrediction)
	print("The accuracy obtain on the final test set is {} at the spectra level".format(acc))
	verif = ""
	if m == realLabels[0]:
		verif = "right prediction"
	else:
		verif = "bad prediction"
	print("The group predicted for this patient is {} and it's a {}".format(pred, verif))
	return acc

def loadModel(path, gpus_list, verbose=1, visualization=False, pathToVisualization=None, numModel=None, CAM=False):
	if visualization and pathToVisualization == None:
		raise ValueError("If visualization is set to True, we need a path where store the result of the visualization")
	model, loss, optimizer, device = createCNN(gpus_list, CAM=CAM)
	checkpoint = torch.load(path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	train_losses = checkpoint['train_loss']
	train_acc = checkpoint['train_acc']
	val_losses = checkpoint['val_loss']
	val_acc = checkpoint['val_acc']
	if visualization:
		modelName = 'Model ' + str(numModel + 1)
		visualizeTrain(modelName, train_losses, val_losses, train_acc, val_acc, pathToVisualization)
	if verbose == 1:
		print("------------------------------ Final result of the model ! ------------------------------")
		print("Train loss : {}; Train accuracy : {}; \n Validation loss : {}; Validation accuracy : {}".format(
			train_losses[-1], train_acc[-1], val_losses[-1], val_acc[-1]))
	return model, device


def trainingLoop(gpus_list, numEpochs, patience, train_dataset, validation_dataset, directoryToSaveModel, verbose=1, visualization=False, pathToVisualization=None, CAM=False):
	if visualization and pathToVisualization == None:
		raise ValueError("If visualization is set to True, we need a path where store the result of the visualization")
	models_list = []
	devices = []
	if not (os.path.exists(directoryToSaveModel)):
		os.mkdir(directoryToSaveModel)
	start = time.time()
	for i in range(len(train_dataset)):
		path = directoryToSaveModel + "/" + str(i + 1) + ".pckl"
		if (os.path.exists(path)):
			print("------------------------------ Model {} already exist, we load it ! ------------------------------".format(i + 1))
			model, device = loadModel(path, gpus_list, CAM=CAM)
			models_list.append(model)
			devices.append(device)
		else:
			print("------------------------------ Let's train model {} ! ------------------------------".format(i + 1))
			model, loss, optimizer, device = createCNN(gpus_list, CAM=CAM)
			train_loss, val_loss, train_acc, val_acc = train(device, model, loss, optimizer, train_dataset[i],
			                                                 validation_dataset[i], numEpochs, patience, path, verbose=verbose)
			if visualization:
				modelName = 'Model ' + str(i + 1)
				visualizeTrain(modelName, train_loss, val_loss, train_acc, val_acc, pathToVisualization)
			devices.append(device)
			models_list.append(model)
	end = time.time()
	hT = (end-start)//3600
	mT = ((end-start)%3600)//60
	sT = (((end-start)%3600)%60)
	print("------------------------------ Total time of training {} h {} m and {} s ------------------------------".format(hT, mT, sT))

	return models_list, devices

def testLoop(models_list, test_dataset, devices, groups, folds, patient_level_prediction, patient_level_real):
	test_accs = []
	groups_test = []
	for i, (train_idx, test_idx) in enumerate(folds):
		groups_test.append(groups[test_idx])

	for i in range(len(test_dataset)):
		print(
			"------------------------------ Let's predict with model {} ! ------------------------------".format(i + 1))
		acc, patient_level_prediction = testModel(models_list[i], test_dataset[i], devices[i], groups_test[i], patient_level_prediction)
		test_accs.append(mean(acc))
		print(
			"------------------------------ Model {} predict with {} of accuracy at the spectra level ------------------------------".format(
				i + 1, mean(acc)))
	total_acc = 0
	for i in range(len(test_accs)):
		total_acc += test_accs[i]
	print("------------------------------ The mean accuracy is {} as the spectra level ------------------------------".format(total_acc / len(test_accs)))

	final_patient_level = {}
	for patient in patient_level_prediction:
		final_patient_level[patient] = patient_level_prediction[patient].index(max(patient_level_prediction[patient]))
	acc_patient_level = accuracy_score(list(patient_level_real.values()), list(final_patient_level.values()))

	print("------------------------------ The accuracy is {} at the patient level ------------------------------".format(acc_patient_level))

	cm = confusion_matrix(list(patient_level_real.values()), list(final_patient_level.values()))
	labels = ["COV +", "COV -", "CTRL"]
	heatmap(cm, cbar=False, annot=True, xticklabels=labels, yticklabels=labels)
	plt.title("Confusion matrix", fontsize =20)
	plt.xlabel('Predicted', fontsize = 15)
	plt.ylabel('Real', fontsize = 15)
	plt.show()