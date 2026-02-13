from load_data import CovidData
from lstm import LSTM
from train_model import train_model
from torch.utils.data import DataLoader

covid_data_obj = CovidData() 
covid_data = DataLoader(covid_data_obj, batch_size=1, shuffle=False)

lstm = LSTM(1, 10, 1, True, 200)
print(f"***Initialized LSTM with {lstm.num_param} parameters***")

# trains the LSTM and saves the parameters in a file
train_model(lstm, 20, covid_data, "trial_lstm.param")
