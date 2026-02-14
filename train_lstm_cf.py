from load_data import CovidData
from lstm import LSTM
from train_model import train_model
from torch.utils.data import DataLoader
import time

covid_data_obj = CovidData() 
covid_data = DataLoader(covid_data_obj, batch_size=1, shuffle=False)

# initialize the LSTM model
lstm = LSTM(1, 10, 1, True, 200)
print(f"***Initialized LSTM with {lstm.num_param} parameters***")

# train the LSTM and save the parameters in a file
stime = time.perf_counter()
train_model(lstm, 20, covid_data, "trial_lstm.param")
etime = time.perf_counter()

print(f"***LSTM finished training in {etime-stime}s***")

# initialize the CF model
cf = CF()
print(f"***Initialized CF with {cf.num_param} parameters***")

# train the CF and save the parameters in a file
stime = time.perf_counter()
train_model(cf, 20, covid_data, "trial_cf.param")
etime = time.perf_counter()

print(f"***CF finished training in {etime-stime}s***")
