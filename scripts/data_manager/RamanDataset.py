import torch
from torch.utils.data import Dataset

class RamanDataset(Dataset):
	def __init__(self, X, Y):
		super(RamanDataset).__init__()
		x = torch.from_numpy(X)
		self.raman_spectra = x
		y = torch.from_numpy(Y)
		self.labels = y

	def __len__(self):
		return len(self.raman_spectra)

	def __getitem__(self, idx):
		spectrum = self.raman_spectra[idx]
		label = self.labels[idx]

		return spectrum, label