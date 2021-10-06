import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# working with the MNIST dataset in this example
# Creating a fully connected Network
class NN(nn.Module):
    # initialization of the model
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    # defining forward pass
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
input_size = 28*28 # the input size of the image is 28*28
num_classes = 10
lr = 1e-3
batch_size = 64
num_epochs = 5

# load dataset
train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# initialize the network
model = NN(input_size, num_classes).to(device)

#loss and optimizer
lossCE = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        # get data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)

        # flatenning the data
        data = data.reshape(data.shape[0], -1)

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
            x = x.to(device)
            y= y.to(device)
            x = x.reshape(x.shape[0],-1)

            logits = model(x)
            _,predictions = logits.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f"Accuracy: {(num_correct/num_samples)*100:.2f}")
    model.train()
accuracy(train_dataloader, model)
accuracy(test_dataloader, model)
