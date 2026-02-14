import torch
from load_data import TrainData
from lstm import LSTM
from cf import CF
from train_model import train_model
from torch.utils.data import DataLoader
import time

train_data_obj = TrainData()
train_data = DataLoader(train_data_obj, batch_size=1, shuffle=True)

# initialize LSTM with saved params
lstm_param_path = "./lstm.param"
lstm = LSTM(1, 30, 1, True, 200)
lstm.load_state_dict(torch.load(lstm_param_path))

# initialize CF with saved params
cf_param_path = "./cf.param"
cf = CF()
cf.load_state_dict(torch.load(cf_param_path))

lstm_time = 0
cf_time = 0
loss_func = torch.nn.MSELoss()

cf_better = 0
lstm_better = 0

with torch.no_grad():
	for batch_num, (inputs, act_outputs) in enumerate(train_data):
		stime = time.perf_counter()
		lstm_out = lstm(inputs)
		etime = time.perf_counter()
		lstm_time += etime-stime
		lstm_loss = loss_func(lstm_out, act_outputs)

		stime = time.perf_counter()
		cf_out = cf(inputs)
		etime = time.perf_counter()
		cf_time += etime-stime
		cf_loss = loss_func(cf_out, act_outputs)

		if cf_loss < lstm_loss:
			cf_better += 1
		else:
			lstm_better += 1

print(f"LSTM took {lstm_time}s overall for inference")
print(f"CF took {cf_time}s overall for inference")
print(f"CF was better on {cf_better/(cf_better+lstm_better)*100}% of the data")
