import torch

class CF(torch.nn.Module):
	def __init__(self):
		super(CF, self).__init__()
		self.conv10 = torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=10)
		self.conv25 = torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=25)
		self.conv50 = torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=50)
		
		self.dnn = torch.nn.Linear(4144, 1)
		self.num_param = 0
		for p in self.parameters():
			self.num_param += p.numel()

	def forward(self, x):
		x1 = self.conv10(x.view([1, 1, -1]))
		x2 = self.conv25(x.view([1, 1, -1]))
		x3 = self.conv50(x.view([1, 1, -1]))
		
		x = torch.cat((torch.flatten(x1), torch.flatten(x2), torch.flatten(x3)))
		x = self.dnn(x)
		return x
