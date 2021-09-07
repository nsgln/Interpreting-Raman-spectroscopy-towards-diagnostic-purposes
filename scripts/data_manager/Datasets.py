from torch.utils.data import Dataset

class Datasets(Dataset):
	def __init__(self, ramanDatasets):
		super(Datasets).__init__()
		self.datasets = ramanDatasets

	def __len__(self):
		return len(self.datasets)

	def __getitem__(self, idx):
		return self.datasets[idx]