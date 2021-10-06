# creating custom datasts for our files
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

"""
A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. 
Take a look at this implementation; the FashionMNIST images are stored in a directory img_dir, 
and their labels are stored separately in a CSV file annotations_file.
"""

# The labels.csv file looks like this
"""
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
"""

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None,
    target_transform = None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# prepping the data to train with dataloaders
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root = "data",
    download = True,
    train = True,
    transform = ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")