import torch

# data_loader must be a torch DataLoader object
# trains the model and stores the final parameters in a file
def train_model(model, num_epochs, data_loader, param_file_path):
	if torch.cuda.is_available():
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")

	print("Using device: ", device)
	
	model = model.to(device)
	optimizer = torch.optim.SGD(model.parameters())
	loss_func = torch.nn.MSELoss()
	
	for e in range(num_epochs):
		for batch_num, (inputs, act_outputs) in enumerate(data_loader):	
			optimizer.zero_grad()
			model_outputs = model(inputs)
			loss = loss_func(model_outputs, act_outputs)
			loss.backward()
			optimizer.step()

		if e%10==9 or e==num_epochs-1:
			print(f"epoch: {e+1} done")

	torch.save(model.state_dict(), param_file_path)
