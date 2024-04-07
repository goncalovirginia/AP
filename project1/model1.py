import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.io import read_image
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import CustomImageDataset as CID
# Load the train dataset
train_dataset = CID.CustomImageDataset(annotations_file='./images/train.csv', img_dir='./images/train/',transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load the test dataset
test_dataset = CID.CustomImageDataset(annotations_file='./images/test.csv', img_dir='./images/test/',transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Model


y_pred = "prediction";

# Save in dataframe for Kaggle submission
submission = pd.DataFrame({'Id': test_dataset.img_labels.iloc[:, 0], 'main_type': y_pred})
submission.to_csv('./submission.csv', index=False)
