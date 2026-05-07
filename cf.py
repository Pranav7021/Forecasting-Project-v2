import torch
import random

class ConvLayer(torch.nn.Module):
	def __init__(self, seq_len, nfeatures, nhead, dropout=0):
		super(ConvLayer, self).__init__()
		
		self.seq_len = seq_len
		self.nfeatures = nfeatures
		self.nhead = nhead
		self.dropout = dropout

		self.convs = torch.nn.ModuleList()
		self.dnns = torch.nn.ModuleList()
		self.max_pool = torch.nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

		for i in range(nhead):
			self.convs.append(torch.nn.Conv1d(in_channels=nfeatures, out_channels=nfeatures*2, kernel_size=(i+2)**2))
			self.dnns.append(torch.nn.Linear(in_features = nfeatures*2*(seq_len-(i+2)**2+1), out_features = nfeatures*seq_len//nhead))

	def forward(self, x):
		nx = []
		for i in range(self.nhead):
			y = self.convs[i](x)
			y = torch.flatten(y)
			y = self.dnns[i](y)
			nx.append(y)

		x = torch.cat(nx).view((1, self.nfeatures, self.seq_len))
		x = self.max_pool(x)
		return x.view((self.nfeatures//2, self.seq_len))

class CF(torch.nn.Module):
	def __init__(self, num_att_layers, dim):
		super(CF, self).__init__()

		if torch.cuda.is_available():
			device = torch.device("cuda")
		elif torch.backends.mps.is_available():
			device = torch.device("mps")
		else:
			device = torch.device("cpu")

		self.dim = dim
		self.emb = torch.nn.Embedding(num_embeddings=100, embedding_dim=dim)
		
		self.num_att_layers = num_att_layers	

		self.conv_layers = torch.nn.ModuleList()

		for i in range(num_att_layers):
			sz = dim//pow(2, i)
			self.conv_layers.append(ConvLayer(seq_len=100, nfeatures=sz, nhead=5))

		self.inds = []
		for i in range(100):
			self.inds.append([i])
		
		self.inds = torch.tensor(self.inds).to(device)

		self.dnn = torch.nn.Linear((self.dim//pow(2, self.num_att_layers))*100, 1)

		self.num_param = 0
		for p in self.parameters():
			self.num_param += p.numel()

	def forward(self, x): # x is (1, 100)
		embeddings = self.emb(self.inds).view((100, self.dim)).T
		x = torch.mul(embeddings, x)

		for i in range(self.num_att_layers):
			x = self.conv_layers[i](x)

		x = torch.flatten(x)
		return self.dnn(x)
