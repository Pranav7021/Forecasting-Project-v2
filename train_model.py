import torch

# data_loader must be a torch DataLoader object
# trains the model and stores the final parameters in a file
def train_model(model, num_epochs, data_loader, param_file_path):
	optimizer = torch.optim.SGD(model.parameters())
	loss_func = torch.nn.MSELoss()
	
	for e in range(num_epochs):
		cur_loss = 0
		for batch_num, (inputs, act_outputs) in enumerate(data_loader):	
			optimizer.zero_grad()
			model_outputs = model(inputs)
			loss = loss_func(model_outputs, act_outputs)
			loss.backward()
			optimizer.step()
			cur_loss += loss.item()

		print(f"epoch: {e}, loss: {cur_loss}")

	torch.save(model.state_dict(), param_file_path)
