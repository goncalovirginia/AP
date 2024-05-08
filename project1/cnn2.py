import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.datasets as datasets
from torchvision.io import read_image
from torchvision import models
from sklearn.metrics import f1_score
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
from torch.nn import Linear, ReLU, LeakyReLU, LayerNorm, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Flatten
from torch.optim import Adam
from torch.autograd import Variable
import torch.utils.data.sampler as sampler

# GPU

USE_GPU = True
device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

# Constants

IMAGES_PATH = "images/"
IMG_DIMENSIONS = (300, 400)
NUM_CLASSES = 18
BATCH_SIZE = 64
BATCH_PRINTING_INTERVAL = 5
NUM_EPOCHS = 30
WEIGHT_DECAY = 0.0001
LEARNING_RATE = 0.0005
MODEL_NAME = 'ResNet101'

# Preprocess

preprocess = transforms.Compose([
    transforms.Resize(size=(100, 100)).to(device),
    transforms.RandomHorizontalFlip(0.5).to(device),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).to(device),
    transforms.RandomAffine(degrees=30, translate=(0.25, 0.25), scale=(0.75, 1.25)).to(device),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).to(device)
]).to(device)

predict_preprocess = transforms.Compose([
    transforms.Resize(size=(100, 100)).to(device),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).to(device),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).to(device)
]).to(device)

# Training and validation datasets

dataset = CID.CustomImageDataset(annotations_file=IMAGES_PATH+'train.csv', img_dir=IMAGES_PATH+'train/', transform=preprocess)
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

print('Class distribution:')
print(dataset.get_class_distribution())
print('Class weights:')
print(dataset.get_class_weights())

class_weights = dataset.get_class_weights().tolist()
class_proportion_multipliers = [0.25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.05, 1.10, 1.15, 1.20, 1.25, 1.25]

for i in range(0, NUM_CLASSES):
    class_weights[i] *= class_proportion_multipliers[i]

sample_weights = [0] * len(train_dataset)

for i, (_, label) in enumerate(train_dataset):
    sample_weights[i] = class_weights[label]

weighted_random_sampler = sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, pin_memory=True, batch_size=BATCH_SIZE, sampler=weighted_random_sampler)#, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)#, num_workers=4)

# Models

class SimpleMLP(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.fully_connected_layers = Sequential(
            Flatten(),
            Linear(360000, 2048),
            ReLU(2048),
            Linear(2048, 1024),
            ReLU(1024),
            Linear(1024, 256),
            ReLU(256),
            Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.fully_connected_layers(x)
        return x

class SimpleCNN(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.convolutional_layers = Sequential(
            LayerNorm(IMG_DIMENSIONS),

            Conv2d(3, 16, 5),
            ReLU(32),
            MaxPool2d(2, 2),
            
            Conv2d(16, 32, 3),
            ReLU(32),
            MaxPool2d(2, 2),
        )

        self.fully_connected_layers = Sequential(
            Flatten(),
            Linear(228928, 256),
            ReLU(256),
            Linear(256, 128),
            ReLU(128),
            Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.fully_connected_layers(x)
        return x

class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.convolutional_layers = Sequential(
            LayerNorm(IMG_DIMENSIONS),

            Conv2d(3, 96, 11, 4),
            ReLU(inplace=True),
            MaxPool2d(3, 2),

            Conv2d(96, 256, 5, 1),
            ReLU(inplace=True),
            MaxPool2d(3, 2),

            Conv2d(256, 384, 3, 1),
            ReLU(inplace=True),

            Conv2d(384, 384, 3, 1),
            ReLU(inplace=True),

            Conv2d(384, 256, 3, 1),
            ReLU(inplace=True),
            MaxPool2d(3, 2),

            Dropout()
        )

        self.fully_connected_layers = Sequential(
            Flatten(),
            Dropout(),
            Linear(7168, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Linear(4096, NUM_CLASSES),
            Softmax(dim=1),
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.fully_connected_layers(x)
        return x

#cnn = SimpleCNN()
#cnn.to(device)

# Pre-trained models

cnn = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
cnn.fc = nn.Linear(cnn.fc.in_features, NUM_CLASSES)
cnn.to(device)

# Loss function and optimizer

criterion = nn.CrossEntropyLoss()
criterion.to(device)

optimizer = optim.Adam(cnn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
#optimizer = optim.RMSprop(cnn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.2, patience=0, verbose=True)
optimizer_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=30, eta_min=0)

# Early stopper

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True
        return False

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

        if batch_index % BATCH_PRINTING_INTERVAL == BATCH_PRINTING_INTERVAL-1:
            avg_loss_across_batches = running_loss / BATCH_PRINTING_INTERVAL
            avg_acc_across_batches = running_accuracy / BATCH_PRINTING_INTERVAL
            running_loss = 0.0
            running_accuracy = 0.0
            print(f'Batch {batch_index+1}, Loss: {avg_loss_across_batches:.3f}, Accuracy: {avg_acc_across_batches:.3f}')

    print()

def validate_epoch():
    cnn.train(False)
    running_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            predictions = cnn(inputs)
            predicted_classes = torch.argmax(predictions, dim=1)

            loss = criterion(predictions, labels) # One number, the average batch loss
            running_loss += loss.item()

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted_classes.cpu().tolist())

    epoch_avg_loss = running_loss / len(validation_loader)
    epoch_f1_score = f1_score(y_true=y_true, y_pred=y_pred, average='micro')

    print(f'Val Loss: {epoch_avg_loss:.3f}, F1 Score: {epoch_f1_score:.3f}\n')
    return epoch_avg_loss, epoch_f1_score

def train():
    print('Training...\n')
    epoch_f1_score = 0.0
    early_stopper = EarlyStopper(patience=3, min_delta=0)

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch + 1}\n')
        train_epoch()
        epoch_loss, epoch_f1_score = validate_epoch()

        #optimizer_scheduler.step(epoch_loss)
        optimizer_scheduler.step()
        
        if early_stopper.early_stop(epoch_loss):
            break

    print('Finished Training\n')
    return epoch_f1_score

# Save model

def save_model(cnn, validation_f1_score):
    torch.save(cnn.state_dict(), f'{MODEL_NAME}-lr{LEARNING_RATE}-wd{WEIGHT_DECAY}-{validation_f1_score}.pth')

# Load model

def load_model(model_name):
    cnn = SimpleCNN()
    cnn.to(device)
    cnn.load_state_dict(torch.load(f'{model_name}'))
    cnn.eval()
    print('Model loaded\n')
    return cnn

# Predict submission and save it to .csv

def predict_submission(model, validation_f1_score):
    predict_dataset = CID.CustomImageDataset(annotations_file=IMAGES_PATH+'test_formatted.csv', img_dir=IMAGES_PATH+'test/', transform=predict_preprocess)
    predict_loader = DataLoader(predict_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    predictedLabelsDataframe = pd.DataFrame(columns=['main_type'])
    cnn.train(False)
    
    print('Predicting...\n')

    with torch.no_grad():
        for i, data in enumerate(predict_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            _, predictions = torch.max(model(inputs).cpu(), 1)

            for prediction in predictions:
                predictedLabelsDataframe.loc[len(predictedLabelsDataframe)] = [prediction.item()]

    idDataframe = pd.read_csv('images/test_formatted.csv').rename(columns={'image_id':'Id'})
    IdMainTypeDataframe = idDataframe.join(predictedLabelsDataframe)
    IdMainTypeDataframe.to_csv(f'{MODEL_NAME}-lr{LEARNING_RATE}-wd{WEIGHT_DECAY}-{validation_f1_score}.csv', index=False)
    print('Finished predictions')

# Run stuff

validation_f1_score = train()
save_model(cnn, validation_f1_score)
#cnn = load_model('CNN2-0.13109756097560976.pth')
predict_submission(cnn, validation_f1_score)
