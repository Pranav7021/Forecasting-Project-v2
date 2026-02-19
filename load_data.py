import torch
import pandas as pd
import numpy as np

class TrainData(torch.utils.data.Dataset):
	def __init__(self):
		self.covid_data_path = "./data/Covid_19_Data.csv"
		self.covid_df = pd.read_csv(self.covid_data_path)
		self.aapl_data_path = "./data/AAPL.csv"
    self.aapl_df = pd.read_csv(self.aapl_data_path)
	
		self.confirmed = pd.Series.to_numpy(self.covid_df["Confirmed"])
		self.confirmed = (self.confirmed - np.mean(self.confirmed)) / np.std(self.confirmed)
		self.close = pd.Series.to_numpy(self.aapl_df["close"])
    self.close = (self.close - np.mean(self.close)) / np.std(self.close)

		self.data_x = []
		self.data_y = []
		self.seq_len = 100

		for i in range(len(self.confirmed) - self.seq_len):
			self.data_x.append(self.confirmed[i: i+self.seq_len])
			self.data_y.append(self.confirmed[i+self.seq_len])

		for i in range(len(self.close) - self.seq_len):
      self.data_x.append(self.close[i: i+self.seq_len])
      self.data_y.append(self.close[i+self.seq_len])

		self.data_x = np.array(self.data_x, dtype=np.float32)
		self.data_y = np.array(self.data_y, dtype=np.float32)
		self.len = np.size(self.data_y)
		print(f"***Initialized training dataset with {self.len} examples***")

	def __len__(self):
		return self.len

	def __getitem__(self, i):
		return self.data_x[i], self.data_y[i]

class StockData(torch.utils.data.Dataset):
  def __init__(self):
    self.nvda_data_path = "./data/NVDA.csv"
    self.nvda_df = pd.read_csv(self.nvda_data_path)

    self.close = pd.Series.to_numpy(self.nvda_df["close"])
    self.close = (self.close - np.mean(self.close)) / np.std(self.close)
    self.data_x = []
    self.data_y = []
    self.seq_len = 100

    for i in range(len(self.close) - self.seq_len):
      self.data_x.append(self.close[i: i+self.seq_len])
      self.data_y.append(self.close[i+self.seq_len])

    self.data_x = np.array(self.data_x, dtype=np.float32)
    self.data_y = np.array(self.data_y, dtype=np.float32)
    self.len = np.size(self.data_y)
    print(f"***Initialized stock dataset with {self.len} examples***")

  def __len__(self):
    return self.len

  def __getitem__(self, i):
    return self.data_x[i], self.data_y[i]
