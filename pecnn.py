import torch
import random

class CF(torch.nn.Module):
	def __init__(self, num_conv):
		super(CF, self).__init__()

		if torch.cuda.is_available():
			device = torch.device("cuda")
		elif torch.backends.mps.is_available():
			device = torch.device("mps")
		else:
			device = torch.device("cpu")

		#self.emb1 = torch.nn.Embedding(num_embeddings=100, embedding_dim=32, max_norm=30)
		#self.emb2 = torch.nn.Embedding(num_embeddings=100, embedding_dim=32, max_norm=30)
		#self.emb3 = torch.nn.Embedding(num_embeddings=100, embedding_dim=32, max_norm=30)
		#self.emb4 = torch.nn.Embedding(num_embeddings=100, embedding_dim=32, max_norm=30)
		self.emb1 = torch.nn.Embedding(num_embeddings=100, embedding_dim=32)
		self.emb2 = torch.nn.Embedding(num_embeddings=100, embedding_dim=32)
		self.emb3 = torch.nn.Embedding(num_embeddings=100, embedding_dim=32)
		self.emb4 = torch.nn.Embedding(num_embeddings=100, embedding_dim=32)
		
		self.num_conv = num_conv
		self.convs = torch.nn.ModuleList()
		self.norms = torch.nn.ModuleList()
		random.seed(1)
		for i in range(num_conv):
			x = random.randint(1, 10)
			y = random.randint(1, 5)
			print("Convolution size: ", x, " ", y)
			self.convs.append(torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(x,y)))
			self.norms.append(torch.nn.BatchNorm2d(4, affine=True))

		self.inds = []
		for i in range(100):
			self.inds.append([i])
		
		self.inds = torch.tensor(self.inds).to(device)
		self.relu = torch.nn.ReLU()

		self.dnn1 = torch.nn.Linear(128000, 10)
		self.dnn2 = torch.nn.Linear(10, 1)
		self.num_param = 0
		for p in self.parameters():
			self.num_param += p.numel()

	def forward(self, x): # x is (1, 100)
		#print("first: ", x)
		e1 = []
		e2 = []
		e3 = []
		e4 = []

		for i in range(100):
			e1.append(x[0][i] + self.emb1(self.inds[i]))
			e2.append(x[0][i] + self.emb2(self.inds[i]))
			e3.append(x[0][i] + self.emb3(self.inds[i]))
			e4.append(x[0][i] + self.emb4(self.inds[i]))

		#print(self.emb1.weight)
		e1 = torch.concat(e1)
		e2 = torch.concat(e2)
		e3 = torch.concat(e3)
		e4 = torch.concat(e4)

		x = torch.stack((e1, e2, e3, e4)).unsqueeze(0)

		nx = []
		for i in range(self.num_conv):
			y = self.convs[i](x)
			#y = self.norms[i](x)
			y = self.relu(x)
			nx.append(torch.flatten(y))

		x = torch.cat(nx)
		#print("third: ", x)

		#print(x.shape)
		x = self.dnn1(x)
		x = self.relu(x)
		#print(x)
		return self.dnn2(x)
