import torch
import torchvision
from torch import nn # this contains all the type of layers like Conv2d, Maxpool2d
from torch import optim # optimization functions like SGD Adam
from torch.utils.data import DataLoader
from torchvision import datasets # datasets like MNIST, FashionMNIST
from torch.nn import functional as F # non parameterized layers like Relu tanh
from torchvision import transforms # transformations associated with the datasets

# set dvevice
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28 # the size of the input vector to each state
sequence_length = 28 # number of states
num_layers = 2 # number of hidden layers in RNN per state
hidden_size = 256 # number of nodes in the hidden layer
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 2

# create a RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size #number of nodes in hidden layer
        self.num_layers = num_layers # number of hidden layers
        # we need not explicitly mention the sequence length while initialization
        # similar to RNN we can use GRU and LSTM with similar syntax
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(nput_size, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # information is taken form all the hidden states
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
    
    def forward(self, x):
        # initalizig h0
        # the shape of the dieenstate is (num_hidden_layers, batch_size, number_of_nodes_in_hidden layer)
        # if we are using LSTM then along with the hidden state we have to initialize the cell state
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward prop
        # _ is the nth hidden state
        # for LSTM we will do the forward prop as follows
        # out, (hn, cn) = self,lstm(x, (h0, c0))
        # no changes are required if you will be using GRU
        out,_ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        # if we plan to use only last hidden state insted of using all the hidden states
        # out = self.fc(out[:,-1,:]) # meaning we are taking out[all the batches, the last hidden state, all the features associated with the last hidden state]
        # if we take only the last hidden state then we have to change our Linear layer in model initialiation: nn.Linear(hidden_sizes, num_classes)
        out = self.fc(out)
        return out

# setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initializing the model
model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
num_classes=num_classes).to(device)

# load dataset
train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#loss and optimizer
lossCE = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        # get data to cuda if possible
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        # forward pass
        logits = model(data)
        #loss computation
        loss = lossCE(logits, targets)

        # gradient computation in back prop
        optimizer.zero_grad()
        loss.backward()

        # weight update
        optimizer.step()

def accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    # setting up the model in evaluation mode
    model.eval()

    # Specifying not to compute the gradients
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device).squeeze(1)
            y= y.to(device)

            logits = model(x)
            _,predictions = logits.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f"Accuracy: {(num_correct/num_samples)*100:.2f}")
    model.train()
accuracy(train_dataloader, model)
accuracy(test_dataloader, model)
