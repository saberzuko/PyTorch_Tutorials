import torch
import torch.nn as nn # All the modules lile Conv2d, Linear, BatchNorm, Lossfunctions
import torch.optim as optim # Optimizers like Adam, SGD
import torch.nn.functional as F # All the layers without parameters, relu, tanh
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# working with the MNIST dataset in this example
# Creating a convolutional neural Network
class CNN(nn.Module):
    # initialization of the model
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3),  stride=(1,1), padding=(1,1)) # same convolution preserves the dimensions of the input
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3),  stride=(1,1), padding=(1,1)) # same convolution preserves the dimensions of the input
        self.fc1 = nn.Linear(16*7*7, num_classes)

    # defining forward pass
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
in_channels = 1 # the input size of the image is 28*28
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
model = CNN(in_channels, num_classes).to(device)

#loss and optimizer
lossCE = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        # get data to cuda if possible
        data = data.to(device)
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
            x = x.to(device)
            y= y.to(device)

            logits = model(x)
            _,predictions = logits.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f"Accuracy: {(num_correct/num_samples)*100:.2f}")
    model.train()
accuracy(train_dataloader, model)
accuracy(test_dataloader, model)
