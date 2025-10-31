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
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.rpu_base import cuda

from aihwkit.simulator.presets import TikiTakaEcRamPreset, TTv2EcRamPreset, ChoppedTTv2EcRamPreset, AGADEcRamPreset
from aihwkit.simulator.tiles.transfer_for_batched_TTv2 import TorchTransferTile
#from aihwkit.simulator.tiles.transfer_for_batched_TTv2_v2 import TorchTransferTile


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
EPOCHS = 50


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

    def __init__(self, rpu_config, t = [1,1,1], init_std = 0.0):
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

        if init_std >= 0:
            batch_size = self.l1.rpu_config.batch_size
            # initialize weights  
            init_w = self.l1.get_weights()[0][0]
            noisy_weights = [torch.normal(mean=init_w, std=init_std) for _ in range(batch_size)]
            params = self.l1.get_hidden_parameters()
            params.update({f'slow_weight_{i+1}': noisy_weights[i] for i in range(batch_size)})
            params.update({f'fast_weight': torch.zeros_like(init_w)})
            self.l1.set_hidden_parameters(params)

            init_w = self.l2.get_weights()[0][0]
            noisy_weights = [torch.normal(mean=init_w, std=init_std) for _ in range(batch_size)]
            params = self.l2.get_hidden_parameters()
            params.update({f'slow_weight_{i+1}': noisy_weights[i] for i in range(batch_size)})
            params.update({f'fast_weight': torch.zeros_like(init_w)})
            self.l2.set_hidden_parameters(params)  

            init_w = self.l3.get_weights()[0][0]
            noisy_weights = [torch.normal(mean=init_w, std=init_std) for _ in range(batch_size)]
            params = self.l3.get_hidden_parameters()
            params.update({f'slow_weight_{i+1}': noisy_weights[i] for i in range(batch_size)})
            params.update({f'fast_weight': torch.zeros_like(init_w)})
            self.l3.set_hidden_parameters(params)
        

    def digital_evaluation(self):
        self.digital_eval = True
        self.training = False

    def analog_evaliation(self):
        self.digital_eval = False
        self.training = False   

    def forward(self, x):
        if self.training:
            x = self.l1(x)
            x = torch.sigmoid(x)
            x = self.l2(x)
            x = torch.sigmoid(x)
            x = self.l3(x)
            x = self.logsoftmax(x)
            return x
        else:
            with torch.no_grad():
                w_l1 = torch.mean(self.l1.get_weights()[0], dim=0)
                w_l2 = torch.mean(self.l2.get_weights()[0], dim=0)
                w_l3 = torch.mean(self.l3.get_weights()[0], dim=0)
                if  self.digital_eval:
                    self.l1_n = nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0], bias=True).to(x.device)
                    self.l1_n.weight.data = w_l1
                    self.l1_n.bias.data = self.l1.get_weights()[1]
                    self.l2_n = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], bias=True).to(x.device)
                    self.l2_n.weight.data = w_l2
                    self.l2_n.bias.data = self.l2.get_weights()[1]
                    self.l3_n = nn.Linear(HIDDEN_SIZES[1], OUTPUT_SIZE, bias=True).to(x.device)
                    self.l3_n.weight.data = w_l3
                    self.l3_n.bias.data = self.l3.get_weights()[1]
                else:
                    self.l1_n = AnalogLinear(INPUT_SIZE, HIDDEN_SIZES[0], bias=True, rpu_config=self.l1.rpu_config).to(x.device)
                    self.l1_n.set_weights(w_l1, bias=self.l1.get_weights()[1])
                    self.l2_n = AnalogLinear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], bias=True, rpu_config=self.l2.rpu_config).to(x.device)
                    self.l2_n.set_weights(w_l2, bias=self.l2.get_weights()[1])
                    self.l3_n = AnalogLinear(HIDDEN_SIZES[1], OUTPUT_SIZE, bias=True, rpu_config=self.l3.rpu_config).to(x.device)
                    self.l3_n.set_weights(w_l3, bias=self.l3.get_weights()[1])

                x = self.l1_n(x)
                x = torch.sigmoid(x)
                x = self.l2_n(x)
                x = torch.sigmoid(x)
                x = self.l3_n(x)
                x = self.logsoftmax(x)
                return x


def train(train_set, validation_dataset, model, optimizer, scheduler, classifier):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
    """
    model.train()
    time_init = time()
    l = []
    a = []
    for epoch_number in range(EPOCHS):
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

            # Run training (backward propagation).
            loss.backward()

            # Optimize weights.
            optimizer.step()

        # Evaluate the trained model.
        loss, acc = test_evaluation(model, validation_dataset, classifier)
        print("Epoch {} - Test loss: {:.3f}".format(epoch_number, loss))
        print("           Accuracy: {:.3f}\n".format(acc))
        l.append(loss)
        a.append(acc)

        # Decay learning rate if needed.
        scheduler.step()

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

    model.digital_evaluation()

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

    num_tiles = [1, 4, 8, 16, 32, 64]
    batch_size = 64
    istd = -1
    sym_index = [-1, 0.0, 0.5, 1, 1.5, 2.0, 2.5]

    # Load datasets.
    train_dataset, validation_dataset = load_images(batch_size)

    for nt in num_tiles:
        lr = 0.05
        for si in sym_index:
            accuracy = []
            for i in range(1):
                print(f'Run {i}')
                # Prepare the model.
                rpu_config = AGADEcRamPreset()
                rpu_config.batch_size = nt
                rpu_config.device.fast_lr = lr * 1
                t = [1, 1, 1]
                if si >= 0:
                    rpu_config.device.gamma_up = si
                    rpu_config.device.gamma_down = si
                model = Analog_Network(rpu_config, t, istd).to(DEVICE)

                classifier = nn.NLLLoss()
                optimizer = AnalogSGD(model.parameters(), lr=lr)
                optimizer.regroup_param_groups(model)
                scheduler = StepLR(optimizer, step_size=40, gamma=0.5)

                # Train the model.
                loss, acc = train(train_dataset, validation_dataset, model, optimizer, scheduler, classifier)


                accuracy.append(acc)
                print(f'Run {i}:\n  Digital Accuracy: {acc[-1]:.2f}')

            print(f'Mean Digital Accuracy: {np.mean(accuracy):.2f} for num tiles {nt} and sym_index {si}')
            results.append({
                'num_tiles': nt,
                'init_std': istd,
                'sym_index': si,
                'digital_accuracies': accuracy,
                'digital_losses': loss,
            })

    # save results to file
    df = pd.DataFrame(results)
    df.to_csv('mnist_agad_sym_test.csv', index=False)