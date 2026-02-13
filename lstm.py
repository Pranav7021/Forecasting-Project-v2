import torch

class LSTM(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, batch_first):
		super(LSTM, self).__init__()
		self.lstm_layer = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
		self.dnn = torch.nn.Linear(seq_len*hidden_size, 1)

		self.num_param = 0
		for p in self.parameters():
			self.num_param += p.numel()

	def forward(self, x):
		x = x.view([1, -1, 1])
		x, _ = self.lstm_layer(x)
		x = torch.flatten(x)
		x = self.dnn(x)
		return x
