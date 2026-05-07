import torch
import pandas as pd
import numpy as np

class TrainData(torch.utils.data.Dataset):
	def __init__(self):
		if torch.cuda.is_available():
			device = torch.device("cuda")
		elif torch.backends.mps.is_available():
			device = torch.device("mps")
		else:
			device = torch.device("cpu")

		self.aapl_data_path = "./data/AAPL.csv"
		self.aapl_df = pd.read_csv(self.aapl_data_path)
		self.spy_data_path = "./data/SPY.csv"
		self.spy_df = pd.read_csv(self.spy_data_path)
	
		self.close1 = pd.Series.to_numpy(self.aapl_df["close"])
		self.close1 = (self.close1 - np.mean(self.close1)) / np.std(self.close1)
		self.close2 = pd.Series.to_numpy(self.spy_df["close"])
		self.close2 = (self.close2 - np.mean(self.close2)) / np.std(self.close2)

		self.data_x = []
		self.data_y = []
		self.seq_len = 100

		for i in range(len(self.close1) - self.seq_len):
			self.data_x.append(self.close1[i: i+self.seq_len])
			self.data_y.append(self.close1[i+self.seq_len])

		for i in range(len(self.close2) - self.seq_len):
			self.data_x.append(self.close2[i: i+self.seq_len])
			self.data_y.append(self.close2[i+self.seq_len])
		
		self.data_x = np.array(self.data_x, dtype=np.float32)
		self.data_x = torch.tensor(self.data_x, device=device)
		self.data_y = np.array(self.data_y, dtype=np.float32)
		self.len = np.size(self.data_y)
		self.data_y = torch.tensor(self.data_y, device=device)
		print(f"***Initialized training dataset with {self.len} examples***")

	def __len__(self):
		return self.len

	def __getitem__(self, i):
		return self.data_x[i], self.data_y[i]

class StockData(torch.utils.data.Dataset):
	def __init__(self):
		if torch.cuda.is_available():
			device = torch.device("cuda")
		elif torch.backends.mps.is_available():
			device = torch.device("mps")
		else:
			device = torch.device("cpu")

		self.nvda_data_path = "./data/NVDA.csv"
		self.nvda_df = pd.read_csv(self.nvda_data_path)
		self.googl_data_path = "./data/GOOGL.csv"
		self.googl_df = pd.read_csv(self.googl_data_path)
		self.pins_data_path = "./data/PINS.csv"
		self.pins_df = pd.read_csv(self.pins_data_path)

		self.close1 = pd.Series.to_numpy(self.nvda_df["close"])
		self.close1 = (self.close1 - np.mean(self.close1)) / np.std(self.close1)

		self.close2 = pd.Series.to_numpy(self.googl_df["close"])
		self.close2 = (self.close2 - np.mean(self.close2)) / np.std(self.close2)

		self.close3 = pd.Series.to_numpy(self.pins_df["close"])
		self.close3 = (self.close3 - np.mean(self.close3)) / np.std(self.close3)

		self.data_x = []
		self.data_y = []
		self.seq_len = 100

		for i in range(len(self.close1) - self.seq_len):
			self.data_x.append(self.close1[i: i+self.seq_len])
			self.data_y.append(self.close1[i+self.seq_len])

		for i in range(len(self.close2) - self.seq_len):
			self.data_x.append(self.close2[i: i+self.seq_len])
			self.data_y.append(self.close2[i+self.seq_len])
		
		for i in range(len(self.close3) - self.seq_len):
			self.data_x.append(self.close3[i: i+self.seq_len])
			self.data_y.append(self.close3[i+self.seq_len])
		
		self.data_x = np.array(self.data_x, dtype=np.float32)
		self.data_x = torch.tensor(self.data_x, device=device)
		self.data_y = np.array(self.data_y, dtype=np.float32)
		self.len = np.size(self.data_y)
		self.data_y = torch.tensor(self.data_y, device=device)
		print(f"***Initialized stock dataset with {self.len} examples***")

	def __len__(self):
		return self.len

	def __getitem__(self, i):
		return self.data_x[i], self.data_y[i]
