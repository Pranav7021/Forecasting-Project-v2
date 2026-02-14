import torch
import pandas as pd
import numpy as np

class AllData(torch.utils.data.Dataset):
	def __init__(self):
		self.covid_data_path = "./Covid_19_Data.csv"
		self.df = pd.read_csv(self.covid_data_path)
		
		self.confirmed = pd.Series.to_numpy(self.df["Confirmed"])
		self.confirmed = (self.confirmed - np.mean(self.confirmed)) / np.std(self.confirmed)
		self.data_x = []
		self.data_y = []

		for i in range(len(self.confirmed) - 200):
			self.data_x.append(self.confirmed[i: i+200])
			self.data_y.append(self.confirmed[i+200])

		self.data_x = np.array(self.data_x, dtype=np.float32)
		self.data_y = np.array(self.data_y, dtype=np.float32)
		self.len = np.size(self.data_y)
		print(f"***Initialized COVID training dataset with {self.len} examples***")

	def __len__(self):
		return self.len

	def __getitem__(self, i):
		return self.data_x[i], self.data_y[i]
