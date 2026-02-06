import torch

class LSTM(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, batch_first):
		super(LSTM, self).__init__()
		self.lstm_layer = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)

	def forward(self, x):
		x = self.lstm_layer(x)
		return x
