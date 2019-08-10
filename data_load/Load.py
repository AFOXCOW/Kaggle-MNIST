from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
class mnist_all(Dataset):
	def __init__(self,csv_file,transform=None):
		self.data_labels = pd.read_csv(csv_file).values
		self.transform = transform
	def __len__(self):
		return len(self.data_labels)
	def __getitem__(self,idx):
		label = self.data_labels[idx][0]
		label = np.array([label])
		img_flatten = self.data_labels[idx][1:]
		img = img_flatten.reshape((28,28))
		img = np.expand_dims(img,axis=0)

		train_sample = {'image':img,'label':label}

		if self.transform:
			train_sample = self.transform(train_sample)
		return train_sample

class mnist_test(Dataset):
	def __init__(self,csv_file,transform=None):
		self.data_labels = pd.read_csv(csv_file).values
		self.transform = transform
	def __len__(self):
		return len(self.data_labels)
	def __getitem__(self,idx):

		img_flatten = self.data_labels[idx]
		img = img_flatten.reshape((28,28))
		img = np.expand_dims(img,axis=0)
		img = torch.from_numpy(img)
		train_sample = {'image':img}

		if self.transform:
			train_sample = self.transform(train_sample)
		return train_sample

class ToTensor(object):
	def  __call__(self,sample):
		img , label = sample['image'],sample['label']
		return {'image':torch.from_numpy(img),'label':torch.LongTensor(label)}
