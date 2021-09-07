import torch
import torch.nn as nn

class ConvNetCAM(nn.Module):
	def __init__(self):
		super(ConvNetCAM, self).__init__()

		self.cnn_layers = nn.Sequential(
			nn.Conv1d(1, 100, kernel_size=100,
			          stride=1, padding_mode='replicate'),
			nn.ReLU(),
			nn.BatchNorm1d(100, eps=0.001, momentum=0.99),
			nn.Conv1d(100, 102, kernel_size=5,
			          stride=2, padding_mode='replicate'),
			nn.ReLU(),
			nn.MaxPool1d(6, stride=3),
			nn.BatchNorm1d(102, eps=0.001, momentum=0.99),
			nn.Conv1d(102, 25, kernel_size=9,
			          stride=5, padding_mode='replicate'),
			nn.ReLU()
        )
	
		self.GAP= nn.AvgPool1d(25)
		
		self.dense_layers = nn.Sequential(
			nn.Linear(25, 3),
			nn.Softmax(dim=1)
		)
		
	def forward(self, x):
		x = x.resize_(x.shape[0], 1, x.shape[1])
		x = self.cnn_layers(x)
		x = self.GAP(x)
		x = torch.flatten(x, 1)
		x = self.dense_layers(x)
		return x