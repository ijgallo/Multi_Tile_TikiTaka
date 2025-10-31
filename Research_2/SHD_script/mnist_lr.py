# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 3: MNIST training.

MNIST training example based on the paper:
https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full

Uses learning rates of η = 0.01, 0.005, and 0.0025
for epochs 0–10, 11–20, and 21–30, respectively.
"""
# pylint: disable=invalid-name, redefined-outer-name

import os
from time import time
import numpy as np
import pandas as pd

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.rpu_base import cuda

from aihwkit.simulator.presets import AGADEcRamPreset, EcRamPresetDevice
#from aihwkit.simulator.tiles.transfer_for_batched_TTv2 import TorchTransferTile
from aihwkit.simulator.tiles.transfer_for_batched_TTv2_v2 import TorchTransferTile


# Check device
# USE_CUDA = 0
# if cuda.is_compiled():
#     USE_CUDA = 1
DEVICE = torch.device("cpu")#"cuda" if USE_CUDA else 

# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data", "DATASET")

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters.
Target_accuracy = 0.97


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

    def __init__(self, rpu_config, t = [1,1,1], max_iter=500, w_init=0.4, max_iter_zero=3000):
        super(Analog_Network, self).__init__()

        rpu_config.device.units_in_mbatch = True
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.device.transfer_every = t[0]
        self.l1 = TorchTransferTile(
            INPUT_SIZE,
            HIDDEN_SIZES[0],
            rpu_config,
            True,
        )
        rpu_config.device.transfer_every = t[1]
        self.l2 = TorchTransferTile(
            HIDDEN_SIZES[0],
            HIDDEN_SIZES[1],
            rpu_config,
            True,
        )
        rpu_config.device.transfer_every = t[2]
        self.l3 = TorchTransferTile(
            HIDDEN_SIZES[1],
            OUTPUT_SIZE,
            rpu_config,
            True,
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

        num_tiles = self.l1.rpu_config.batch_size
        # initialize weights with xavier
        init_w = torch.nn.init.xavier_normal_(self.l1.get_weights()[0][0])
        #init_w = torch.nn.init.kaiming_normal_(self.l1.get_weights()[0][0])
        self.init_w1 = init_w
        self.l1.tile.program_weights_from_one([init_w for _ in range(num_tiles)], max_iter=max_iter, w_init=w_init, 
                                               max_iter_zero=max_iter_zero)

        init_w = torch.nn.init.xavier_normal_(self.l2.get_weights()[0][0])
        #init_w = torch.nn.init.kaiming_normal_(self.l2.get_weights()[0][0])
        self.init_w2 = init_w
        self.l2.tile.program_weights_from_one([init_w for _ in range(num_tiles)], max_iter=max_iter, w_init=w_init,
                                                  max_iter_zero=max_iter_zero)

        init_w = torch.nn.init.xavier_normal_(self.l3.get_weights()[0][0])
        #init_w = torch.nn.init.kaiming_normal_(self.l3.get_weights()[0][0])
        self.init_w3 = init_w
        self.l3.tile.program_weights_from_one([init_w for _ in range(num_tiles)], max_iter=max_iter, w_init=w_init,
                                                    max_iter_zero=max_iter_zero)
        

    def digitize_model(self):
        w_l1 = torch.mean(self.l1.tile.read_weights(), dim=0) * self.l1.get_scales().item()
        w_l2 = torch.mean(self.l2.tile.read_weights(), dim=0) * self.l2.get_scales().item()
        w_l3 = torch.mean(self.l3.tile.read_weights(), dim=0) * self.l3.get_scales().item()
        self.l1_n = nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0], bias=True)
        self.l1_n.weight.data = w_l1
        self.l1_n.bias.data = self.l1.get_weights()[1]
        self.l2_n = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], bias=True)
        self.l2_n.weight.data = w_l2
        self.l2_n.bias.data = self.l2.get_weights()[1]
        self.l3_n = nn.Linear(HIDDEN_SIZES[1], OUTPUT_SIZE, bias=True)
        self.l3_n.weight.data = w_l3
        self.l3_n.bias.data = self.l3.get_weights()[1]

    def forward(self, x):
        if self.training:
            x = self.l1(x)
            x = torch.sigmoid(x)
            #x = nn.LeakyReLU()(x)
            x = self.l2(x)
            x = torch.sigmoid(x)
            #x = nn.LeakyReLU()(x)
            x = self.l3(x)
            x = self.logsoftmax(x)
            return x 
        else:
            with torch.no_grad():
                x = self.l1_n(x)
                x = torch.sigmoid(x)
                #x = nn.LeakyReLU()(x)
                x = self.l2_n(x)
                x = torch.sigmoid(x)
                #x = nn.LeakyReLU()(x)
                x = self.l3_n(x)
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
    epoch_number = 0
    while len(a) == 0 or a[-1] < Target_accuracy:
        model.train()
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

        # Evaluate the trained model.
        model.digitize_model()
        loss, acc = test_evaluation(model, validation_dataset, classifier)
        print("Epoch {} -  Accuracy: {:.3f}".format(epoch_number, acc))
        l.append(loss)
        a.append(acc)

        # Decay learning rate if needed.
        #scheduler.step()
        epoch_number += 1  
        if l[-1] != l[-1]:
            break

        if epoch_number > 600:
            print("Stopping training after 600 epochs")
            break

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

    nt = 8
    lrs = [0.5, 0.05]
    batch_size = 64
    bls = [2,5,10,31,100]

    # Load datasets.
    train_dataset, validation_dataset = load_images(batch_size)

    for lr in lrs:
        for bl in bls:
            # Run multiple times to average results.
            accuracy = []
            for i in range(1):
                print(f'Tiles {nt}, BL {bl}, Run {i}')
                # Prepare the model.
                rpu_config = AGADEcRamPreset()
                rpu_config.device.unit_cell_devices = [EcRamPresetDevice(
                    gamma_up = -0.38,
                    gamma_down = 0.76,
                    allow_increasing=True
                ) for _ in range(2)]
                rpu_config.batch_size = nt
                rpu_config.update.desired_bl = bl
                rpu_config.device.fast_lr = lr * 1
                t = [1, 1, 1]
                model = Analog_Network(rpu_config=rpu_config, t=t, max_iter=20000, w_init=0.4, max_iter_zero=100000).to(DEVICE)

                classifier = nn.NLLLoss()
                optimizer = AnalogSGD(model.parameters(), lr=lr)
                optimizer.regroup_param_groups(model)
                scheduler = StepLR(optimizer, step_size=40, gamma=0.5)

                # Train the model.
                loss, acc = train(train_dataset, validation_dataset, model, optimizer, scheduler, classifier)
                accuracy.append(acc)

            results.append({
                'num_tiles': nt,
                'lr': lr,
                'bl': bl,
                'digital_accuracies': accuracy,
            })

            # save results to file
            df = pd.DataFrame(results)
            df.to_csv('SHD_script/mnist_lr_v2.csv', index=False)