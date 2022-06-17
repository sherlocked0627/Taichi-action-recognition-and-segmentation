import os
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data

from torch.utils.data import Dataset, DataLoader

from config import get_args
args = get_args()

class YangDataset(Dataset):
	def __init__(self, data_type): # data_type is in ['train', 'val', 'test']
		self.data = torch.from_numpy(np.load(getattr(args, data_type + '_path')))
		self.label = torch.from_numpy(np.load(getattr(args, data_type + '_label_path')))
		self.data = self.data.permute(0,2,3,1)
		self.data = torch.unsqueeze(self.data, 4)
		self.label = torch.squeeze(self.label, 0)
		self.label = torch.tensor(self.label, dtype=torch.long)
		self.length = len(self.label)

	def __getitem__(self, index):
		return self.data[index], self.label[index]
	
	def __len__(self):
		return self.length