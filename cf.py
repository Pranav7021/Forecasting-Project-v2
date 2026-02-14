import torch

class CF(torch.nn.Module):
	def __init__(self):
		super(CF, self).__init__()
		self.conv10 = torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=10)
		self.conv25 = torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=25)
		self.conv50 = torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=50)
		self.max_pool = torch.nn.MaxPool1d(kernel_size=3, stride=2)
    	self.conv_on_10 = torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=10)
    	self.conv_on_25 = torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=20)
    	self.conv_on_50 = torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=30)
		
		self.dnn = torch.nn.Linear(1552, 1)
		self.num_param = 0
		for p in self.parameters():
			self.num_param += p.numel()

	def forward(self, x):
		x = x.view([1, 1, -1])
		x1 = self.max_pool(self.conv10(x))
    	x2 = self.max_pool(self.conv25(x))
    	x3 = self.max_pool(self.conv50(x))

    	x1 = self.max_pool(self.conv_on_10(x1))
    	x2 = self.max_pool(self.conv_on_25(x2))
    	x3 = self.max_pool(self.conv_on_50(x3))
		
		x = torch.cat((torch.flatten(x1), torch.flatten(x2), torch.flatten(x3)))
		x = self.dnn(x)
		return x
