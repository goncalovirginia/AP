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
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Flatten
from torch.optim import Adam

# Settings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

# Constants

IMAGES_PATH = "project1/images/"
IMG_DIMENSIONS = (300, 400)
NUM_LABELS = 18
BATCH_SIZE = 32
NUM_EPOCHS = 2

# Preprocess

preprocess = transforms.Compose([
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Training and validation datasets

dataset = CID.CustomImageDataset(annotations_file=IMAGES_PATH+'train.csv', img_dir=IMAGES_PATH+'train/', transform=preprocess)
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)#, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)#, num_workers=4)

# Model

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.convolutional_layers = Sequential(
            Conv2d(3, 96, 9, 4),
            ReLU(inplace=True),
            MaxPool2d(3, 2),

            Conv2d(96, 256, 5, padding=2),
            ReLU(inplace=True),
            MaxPool2d(3, 2),
            
            Conv2d(256, 384, 3, padding=1),
            ReLU(inplace=True),

            Conv2d(384, 384, 3, padding=1),
            ReLU(inplace=True),

            Conv2d(384, 256, 3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(3, 2),

            Dropout(0.2)
        )

        self.fully_connected_layers = Sequential(
            Flatten(),
            Linear(22528, 4096),
            ReLU(inplace=True),
            Dropout(0.2),
            Linear(4096, 2048),
            Linear(2048, NUM_LABELS),
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.fully_connected_layers(x)
        return x

cnn = CNN()
cnn.to(device)

# Loss function and optimizer

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
optimizer = optim.SGD(cnn.parameters(), lr=0.00001, momentum=0.9)

# Training

def train_epoch():
    cnn.train(True)
    running_loss = 0.0
    running_accuracy = 0.0
    
    for batch_index, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = cnn(inputs)
        correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
        running_accuracy += correct / BATCH_SIZE

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_index % 5 == 4:
            avg_loss_across_batches = running_loss / 5
            avg_acc_across_batches = (running_accuracy / 5)
            print('Batch {0}, Loss: {1:.3f}, Accuracy: {2:.3f}'.format(batch_index+1, avg_loss_across_batches, avg_acc_across_batches))
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
    avg_acc_across_batches = running_accuracy / len(validation_loader)

    print('Val Loss: {0:.3f}, Val Accuracy: {1:.3f}\n'.format(avg_loss_across_batches, avg_acc_across_batches))
    return avg_acc_across_batches

def train():
    print('Training...\n')
    epoch__val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch + 1}\n')
        train_epoch()
        epoch__val_acc = validate_epoch()

    print('Finished Training\n')
    return epoch__val_acc

# Save model

def save_model(cnn, accuracy):
    torch.save(cnn.state_dict(), f'project1/saved_models/CNN2-{accuracy}.pth')

# Load model

def load_model(model_name):
    cnn = CNN()
    cnn.to(device)
    cnn.load_state_dict(torch.load(f'project1/saved_models/{model_name}'))
    cnn.eval()
    print('Model loaded\n')
    return cnn

# Predict submission

def predict_submission(model):
    predict_dataset = CID.CustomImageDataset(annotations_file=IMAGES_PATH+'test_formatted.csv', img_dir=IMAGES_PATH+'test/', transform=preprocess)
    predict_loader = DataLoader(predict_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    predictedLabelsDataframe = pd.DataFrame(columns=['main_type'])
    cnn.train(False)
    
    print('Predicting...')

    with torch.no_grad():
        for i, data in enumerate(predict_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            _, predictions = torch.max(model(inputs).cpu(), 1)

            for prediction in predictions:
                predictedLabelsDataframe.loc[len(predictedLabelsDataframe)] = [prediction.item()]

    idDataframe = pd.read_csv('project1/images/test_formatted.csv').rename(columns={'image_id':'Id'})
    IdMainTypeDataframe = idDataframe.join(predictedLabelsDataframe)
    IdMainTypeDataframe.to_csv('project1/predictions/submission1.csv', index=False)
    print('Finished predictions')

# Run stuff

val_accuracy = train()
save_model(cnn, val_accuracy)
#cnn = load_model('CNN2-0.13109756097560976.pth')
predict_submission(cnn)
