import torch
from load_data import TrainData
from lstm import LSTM
from cf import CF
from train_model import train_model
from torch.utils.data import DataLoader
import time

stock_data_obj = StockData()
stock_data = DataLoader(stock_data_obj, batch_size=1, shuffle=False)

# initialize LSTM with saved params
lstm_param_path = "./lstm.param"
lstm = LSTM(1, 30, 1, True, 100)
lstm.load_state_dict(torch.load(lstm_param_path))

# initialize CF with saved params
cf_param_path = "./cf.param"
cf = CF()
cf.load_state_dict(torch.load(cf_param_path))

lstm_time = 0
cf_time = 0
abs_loss = torch.nn.L1Loss(reduction='sum')

cf_better = 0
lstm_better = 0
cf_abs_loss = 0
lstm_abs_loss = 0
combined_abs_loss = 0

with torch.no_grad():
	for batch_num, (inputs, act_outputs) in enumerate(stock_data):
		stime = time.perf_counter()
    lstm_out = lstm(inputs)
    etime = time.perf_counter()
    lstm_time += etime-stime
    lstm_loss = abs_loss(lstm_out, act_outputs)
    lstm_abs_loss += lstm_loss

    stime = time.perf_counter()
    cf_out = cf(inputs)
    etime = time.perf_counter()
    cf_time += etime-stime
    cf_loss = abs_loss(cf_out, act_outputs)
    cf_abs_loss += cf_loss

    combined_out = (lstm_out+cf_out)/2
    combined_loss = abs_loss(combined_out, act_outputs)
    combined_abs_loss += combined_loss

    if cf_loss < lstm_loss:
      cf_better += 1
    else:
      lstm_better += 1

print(f"LSTM took {lstm_time}s overall for inference")
print(f"CF took {cf_time}s overall for inference")
print(f"CF total absolute loss: {cf_abs_loss} and LSTM total absolute loss: {lstm_abs_loss}")
print(f"Combined total absolute loss: {combined_abs_loss}")
print(f"CF was better on {cf_better/(cf_better+lstm_better)*100}% of the data")
