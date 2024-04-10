import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.io import read_image
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import CustomImageDataset as CID
from CustomImageDataset import imshow
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import torch.utils.data

# Settings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Constants

IMAGES_PATH = "project1/images/"
BATCH_SIZE = 8
IMG_DIMENSIONS = (300, 400)
NUM_LABELS = 18
NUM_EPOCHS = 2

# Preprocess

preprocess = transforms.Compose([
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Training and validation datasets

train_dataset = CID.CustomImageDataset(annotations_file=IMAGES_PATH+'train.csv', img_dir=IMAGES_PATH+'train/', transform=preprocess)
train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# Predict dataset

#predict_dataset = CID.CustomImageDataset(annotations_file=IMAGES_PATH+'test_formatted.csv', img_dir=IMAGES_PATH+'test/', transform=preprocess)
#predict_loader = DataLoader(predict_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# Model

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=842240, out_features=1024)
        self.drop1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.drop2 = nn.Dropout(p=0.2)

        self.out = nn.Linear(in_features=512, out_features=18)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.drop2(x)

        x = self.out(x)

        return x
    
cnn = CNN()
cnn.to(device)

# Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.0001)

# Training

def train_epoch():
    cnn.train(True)
    running_loss = 0.0
    running_accuracy = 0.0

    for batch_index, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = cnn(inputs) # shape: [batch_size, 10]
        correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
        running_accuracy += correct / BATCH_SIZE

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_index % 5 == 4:  # print every 5 batches
            avg_loss_across_batches = running_loss / 5
            avg_acc_across_batches = (running_accuracy / 5) * 100
            print('Batch {0}, Loss: {1:.3f}, Accuracy: {2:.1f}%'.format(batch_index+1, avg_loss_across_batches, avg_acc_across_batches))
            running_loss = 0.0
            running_accuracy = 0.0

    print()

def validate_epoch():
    cnn.train(False)
    running_loss = 0.0
    running_accuracy = 0.0

    for i, data in enumerate(validation_loader):
        inputs, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            outputs = cnn(inputs)
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            running_accuracy += correct / BATCH_SIZE
            loss = criterion(outputs, labels) # One number, the average batch loss
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(validation_loader)
    avg_acc_across_batches = (running_accuracy / len(validation_loader)) * 100

    print('Val Loss: {0:.3f}, Val Accuracy: {1:.1f}%\n'.format(avg_loss_across_batches, avg_acc_across_batches))

print('Training...\n')

for epoch in range(NUM_EPOCHS):
    print(f'Epoch: {epoch + 1}\n')
    train_epoch()
    validate_epoch()

print('Finished Training\n')