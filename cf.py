import torch

class CF(torch.nn.Module):
	def __init__(self):
		super(CF, self).__init__()
		self.conv5 = torch.nn.Conv1d(in_channels=1, out_channels=24, kernel_size=5)
		self.conv10 = torch.nn.Conv1d(in_channels=1, out_channels=12, kernel_size=10)
		self.conv25 = torch.nn.Conv1d(in_channels=1, out_channels=12, kernel_size=25)
		self.conv50 = torch.nn.Conv1d(in_channels=1, out_channels=12, kernel_size=50)
		self.max_pool1 = torch.nn.MaxPool1d(kernel_size=3, stride=1)

		self.conv_on_5 = torch.nn.Conv1d(in_channels=24, out_channels=48, kernel_size=5)
		self.conv_on_10 = torch.nn.Conv1d(in_channels=12, out_channels=24, kernel_size=10)
		self.conv_on_25 = torch.nn.Conv1d(in_channels=12, out_channels=24, kernel_size=15)
		self.conv_on_50 = torch.nn.Conv1d(in_channels=12, out_channels=24, kernel_size=20)
		self.max_pool2 = torch.nn.MaxPool1d(kernel_size=4, stride=2)

		self.dnn = torch.nn.Linear(768, 1)
		self.num_param = 0
		for p in self.parameters():
			self.num_param += p.numel()

	def forward(self, x):
		x1 = self.conv5(x[0,80:].view([1,1,-1]))
		x2 = self.max_pool1(self.conv10(x[0,70:].view([1,1,-1])))
		x3 = self.max_pool1(self.conv25(x[0,50:].view([1,1,-1])))
		x4 = self.max_pool1(self.conv50(x.view([1, 1, -1])))

		x1 = self.max_pool2(self.conv_on_5(x1))
		x2 = self.max_pool2(self.conv_on_10(x2))
		x3 = self.max_pool2(self.conv_on_25(x3))
		x4 = self.max_pool2(self.conv_on_50(x4))
		
		x = torch.cat((torch.flatten(x1), torch.flatten(x2), torch.flatten(x3), torch.flatten(x4)))
		x = self.dnn(x)
		return x
