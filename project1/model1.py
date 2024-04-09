import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.io import read_image
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import CustomImageDataset as CID
from CustomImageDataset import imshow
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Constants

IMAGES_PATH = "project1/images/"
BATCH_SIZE = 8
IMG_DIMENSIONS = (300, 400)
NUM_LABELS = 16

# Preprocess

preprocess = transforms.Compose([
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Training dataset

train_dataset = CID.CustomImageDataset(annotations_file=IMAGES_PATH+'train.csv', img_dir=IMAGES_PATH+'train/', transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Testing dataset

test_dataset = CID.CustomImageDataset(annotations_file=IMAGES_PATH+'test.csv', img_dir=IMAGES_PATH+'test/', transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Show some images

dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print('Batch labels:')
print(' '.join(f'{labels[j]}    ' for j in range(BATCH_SIZE)))

# CNN Model

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 292, 9)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(292, 284, 9)
        self.fc1 = nn.Linear(284 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, NUM_LABELS)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

cnn = CNN()

# Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

# Training

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save model

current_datetime = datetime.now()
torch.save(cnn.state_dict(), 'project1/saved_models/CNN-{current_datetime}.pth')

# Predictions

y_predict = "y_predict";

# Save y_predict in .csv

submission = pd.DataFrame({'Id': test_dataset.img_labels.iloc[:, 0], 'main_type': y_predict})
submission.to_csv('project1/predictions/submission-{current_datetime}.csv', index=False)
