import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

df = pd.read_csv('/Users/lucasmichaud/Desktop/Project1/Task1/TrainingData.txt')  # import the data
t = torch.tensor(df['t'].values)
T = torch.tensor([df['tf0'].values, df['ts0'].values])
opt_type = "ADAM"
n = 2  # number of hidden layers
neurons_number = 20  # number of neurons
batch_size = 5
training_set = DataLoader(torch.utils.data.TensorDataset(t, T), batch_size=batch_size, shuffle=True)


# Create the neural Network : input time, output Tf and Ts

class NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons):
        super(NeuralNet, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = nn.Tanh()

        # creation of the network
        if self.hidden_layers != 0:
            self.input_layer = nn.Linear(self.input_dimension, self.neurons)
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
            self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        else:
            print("Simple Linear regression")
            self.linear_regression_layer = nn.Linear(self.input_dimension, self.output_dimension)

    def forward(self, t):
        if self.n_hidden_layers != 0:
            t = self.activation(self.input_layer(t))
            for k, l in enumerate(self.hidden_layers(t)):
                t = self.activation(l(t))
            return self.output_layer
        else:
            return self.linear_regression_layer(t)


# Model definition
network = NeuralNet(input_dimension=t.shape[1], output_dimension=T.shape[1], n_hidden_layers=n, neurons=neurons_number)


# weight initialisation

def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)

    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)

    model.apply(init_weights)


retrain = 1456  # random seed for weight initialisation

init_xavier(network, retrain)

if opt_type == "ADAM":
    optimizer_ = optim.Adam(network.parameters(), lr=0.001)
else:
    raise ValueError("Opt not recognized")


def fit(model, training_set, num_epochs, optimizer, p, verbose=True):
    history = list()

    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose: print("##################", epoch, "###################")

        running_loss = list([0])

        # loop pver batches
        for j, (t_train, T_train) in enumerate(training_set):
            def closure():
                optimizer.zero_grad()
                T_pred = model(t_train)
                loss = torch.mean((T_pred.reshape(-1, ) - T_train.reshape(-1, )) ** p)
                loss.backward()

                running_loss[0] += loss.item()

                return loss

        optimizer.step(closure=closure)
    if verbose: print('Loss: ', (running_loss[0] / len(training_set)))
    history.append(running_loss[0])

    return history

n_epochs = 2000

history = fit(network, training_set, n_epochs, optimizer, p=2, verbose=False)

