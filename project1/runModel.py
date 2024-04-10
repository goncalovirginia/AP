import torch
from torch.utils.data import DataLoader
import CustomImageDataset as CID
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd

#pd.read_csv('project1/images/test.csv', delimiter=';')['image_id'].to_csv('project1/images/test_formatted.csv', index=False)

# Settings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Constants

IMAGES_PATH = "project1/images/"
BATCH_SIZE = 8
IMG_DIMENSIONS = (300, 400)
NUM_LABELS = 18

# Preprocess

preprocess = transforms.Compose([
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Testing dataset

test_dataset = CID.CustomImageDataset(annotations_file=IMAGES_PATH+'test_formatted2.csv', img_dir=IMAGES_PATH+'test/', transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# CNN Model

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 256, 9)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(256, 512, 9)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512 * 69 * 94, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, NUM_LABELS)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)

        return x

# Load model

cnn = CNN()
cnn.to(device)
cnn.load_state_dict(torch.load('project1/saved_models/CNN-{current_datetime}.pth'))
cnn.eval()

# Predict

print('Predicting...')

outputsDataframe = pd.DataFrame(columns=['main_type'])

for i, data in enumerate(test_loader):
    inputs, labels = data[0].to(device), data[1].to(device)
        
    with torch.no_grad():
        outputs = cnn(inputs)

        for output in outputs.cpu().data.numpy():
            outputsDataframe.loc[len(outputsDataframe)] = [output]

outputsDataframe.to_csv('project1/predictions/predictions1.csv', index=False)

print('Predictions complete.')

# Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.0001)

# Validation

def validate_one_epoch():
    cnn.train(False)
    running_loss = 0.0
    running_accuracy = 0.0
    
    for i, data in enumerate(test_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        with torch.no_grad():
            outputs = cnn(inputs) # shape: [batch_size, 10]
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            running_accuracy += correct / BATCH_SIZE
            loss = criterion(outputs, labels) # One number, the average batch loss
            running_loss += loss.item()
        
    avg_loss_across_batches = running_loss / len(test_loader)
    avg_acc_across_batches = (running_accuracy / len(test_loader)) * 100
    
    print('Val Loss: {0:.3f}, Val Accuracy: {1:.1f}%'.format(avg_loss_across_batches,
                                                            avg_acc_across_batches))
    print('***************************************************')
    print()

for epoch_index in range(2):
    print(f'Epoch: {epoch_index + 1}\n')
    validate_one_epoch()