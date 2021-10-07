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

# create a bi-directional LSTM
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).init()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
        bidirectional=True)
        self.fc = self.Linear(hidden_size*2, num_classes)
    
    def forward(self, x):
        # LSTM have both the cell state and the hidden state
        # we are doing number of hidden layers*2 because we are going in both the directions
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size) # the shape of the hidden state is [num_layers*2, batch_size, nodes_in_hiddenLayer]
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:,-1,:])
        return out

# setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initializing the model
model = BiLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
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
