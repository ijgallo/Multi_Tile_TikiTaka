
import os
from time import time
import numpy as np
import pandas as pd

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import copy
import math
# cuda
from torch.backends import cuda

# Check device
DEVICE = torch.device("cuda")# if USE_CUDA else 

# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data", "DATASET")

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters.
EPOCHS = 30


def load_images(batch_size):
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the images.
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    return train_data, validation_data


class Analog_Network(nn.Module):

    def __init__(self):
        super(Analog_Network, self).__init__()

        self.l1 = torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0], bias=True)
        self.l2 = torch.nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], bias=True)
        self.l3 = torch.nn.Linear(HIDDEN_SIZES[1], OUTPUT_SIZE, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # initialize weights with xavier
        torch.nn.init.xavier_normal_(self.l1.weight)
        torch.nn.init.xavier_normal_(self.l2.weight)
        torch.nn.init.xavier_normal_(self.l3.weight)
        # initialize bias with 0
        torch.nn.init.zeros_(self.l1.bias)
        torch.nn.init.zeros_(self.l2.bias)
        torch.nn.init.zeros_(self.l3.bias)

    def forward(self, x):
        x = self.l1(x)
        x = torch.sigmoid(x)
        #x = nn.LeakyReLU()(x)
        x = self.l2(x)
        x = torch.sigmoid(x)
        #x = nn.LeakyReLU()(x)
        x = self.l3(x)
        x = self.logsoftmax(x)
        return x 
            


def train(train_set, validation_dataset, model, optimizer, scheduler, classifier):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
    """
    time_init = time()
    l = []
    a = []

    for epoch_number in range(EPOCHS):
        model.train()
        print("Epoch {}".format(epoch_number))
        for images, labels in train_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # Flatten MNIST images into a 784 vector.
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            # Add training Tensor to the model (input).
            output = model(images)
            loss = classifier(output, labels)
            
            loss.backward()

            # Optimize weights.
            optimizer.step()

        # Decay learning rate if needed.
        #scheduler.step()

        loss, acc = test_evaluation(model, validation_dataset, classifier)
        print("Epoch {} - Test loss: {:.3f}".format(epoch_number, loss))
        print("           Accuracy: {:.3f}\n".format(acc))

    print("\nTraining Time (s) = {}".format(time() - time_init))
    return l, a


def test_evaluation(model, val_set, classifier):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    # Setup counter of images predicted to 0.
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in val_set:
        # Predict image.
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        images = images.view(images.shape[0], -1)
        pred = model(images)

        loss = classifier(pred, labels)

        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()

    accuracy = predicted_ok / total_images
    loss = total_loss / total_images
    return loss, accuracy

if __name__ == "__main__":

    results = []
    batch_size = 64

    # Load datasets.
    train_dataset, validation_dataset = load_images(batch_size)

    lr = 0.3
    # Run multiple times to average results.
    ma_acc = []
    for i in range(1):
        print(f'Run {i}')

        model = Analog_Network().to(DEVICE)
        classifier = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)


        scheduler = StepLR(optimizer, step_size=40, gamma=0.5)

        # Train the model.
        loss, acc = train(train_dataset, validation_dataset, model, optimizer, scheduler, classifier)
        ma_acc.append(test_evaluation(model, validation_dataset, classifier)[1])

        model_average_accuracy = np.mean(ma_acc)
        print(f'    Mean Digital Accuracy: {model_average_accuracy}')