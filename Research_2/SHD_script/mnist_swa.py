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

import copy
import math

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.rpu_base import cuda

from aihwkit.simulator.presets import TikiTakaEcRamPreset, TTv2EcRamPreset, ChoppedTTv2EcRamPreset, AGADEcRamPreset, EcRamPresetDevice
from aihwkit.simulator.tiles.transfer_for_batched_TTv2_log import TorchTransferTile
#from aihwkit.simulator.tiles.transfer_for_batched_TTv2 import TorchTransferTile


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
EPOCHS = 80


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
            

import copy
import math
import torch
import torch.nn as nn

class SWATracker:
    """Stochastic Weight Averaging tracker for aihwkit analog models."""
    
    def __init__(self, model, swa_start=40, swa_freq=1):
        """
        Args:
            model: The Analog_Network model
            swa_start: Epoch to start collecting weights for SWA
            swa_freq: Frequency (in epochs) to collect weights
            input_size: Input dimension size (required for creating averaged model)
            hidden_sizes: List of hidden layer sizes (required for creating averaged model)
            output_size: Output dimension size (required for creating averaged model)
        """
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        
        # Storage for standard SWA weights (unweighted averaging)
        self.swa_weights = {
            'l1': {'weight': None, 'bias': None, 'count': 0},
            'l2': {'weight': None, 'bias': None, 'count': 0},
            'l3': {'weight': None, 'bias': None, 'count': 0}
        }
        
        # Storage for performance-weighted SWA weights
        self.weighted_swa_weights = {
            'l1': {'weight': None, 'bias': None, 'count': 0},
            'l2': {'weight': None, 'bias': None, 'count': 0},
            'l3': {'weight': None, 'bias': None, 'count': 0}
        }
        
        # For performance-weighted averaging
        self.n_tiles = model.l1.rpu_config.batch_size  # Number of tiles in batch
        self.tile_losses = [0.0] * self.n_tiles  # Running losses for each tile
        self.loss_update_count = [0] * self.n_tiles  # To track how many updates each tile received
        
    def update(self, epoch, collect_performance=True, temperature=0.2):
        """Update SWA weights if conditions are met."""
        if epoch < self.swa_start:
            return
            
        if epoch % self.swa_freq != 0:
            return
        
        # If we're collecting performance metrics, also update the weighted average
        if collect_performance and sum(self.loss_update_count) > 0:
            # Get current weights from all tiles
            l1_weights, l1_bias = self.model.l1.tile.read_weights() * self.model.l1.get_scales().item(), self.model.l1.get_weights()[1]
            l2_weights, l2_bias = self.model.l2.tile.read_weights() * self.model.l2.get_scales().item(), self.model.l2.get_weights()[1]
            l3_weights, l3_bias = self.model.l3.tile.read_weights() * self.model.l3.get_scales().item(), self.model.l3.get_weights()[1]

            # Get current weights from all tiles
            # l1_weights, l1_bias = self.model.l1.read_weights(apply_weight_scaling=True)
            # l2_weights, l2_bias = self.model.l2.read_weights(apply_weight_scaling=True)
            # l3_weights, l3_bias = self.model.l3.read_weights(apply_weight_scaling=True)

            self._update_weighted_swa(l1_weights, l1_bias, l2_weights, l2_bias, l3_weights, l3_bias, temperature)
        else:
            # Get current weights from all tiles
            l1_weights, l1_bias = self.model.l1.tile.read_weights_multi() * self.model.l1.get_scales().item(), self.model.l1.get_weights()[1]
            l2_weights, l2_bias = self.model.l2.tile.read_weights_multi() * self.model.l2.get_scales().item(), self.model.l2.get_weights()[1]
            l3_weights, l3_bias = self.model.l3.tile.read_weights_multi() * self.model.l3.get_scales().item(), self.model.l3.get_weights()[1]

            # Get current weights from all tiles
            # l1_weights, l1_bias = self.model.l1.read_weights_multi(apply_weight_scaling=True)
            # l2_weights, l2_bias = self.model.l2.read_weights_multi(apply_weight_scaling=True)
            # l3_weights, l3_bias = self.model.l3.read_weights_multi(apply_weight_scaling=True)
        
        # Update standard SWA (unweighted average across tiles)
        self._update_unweighted_swa(l1_weights, l1_bias, l2_weights, l2_bias, l3_weights, l3_bias)
    
    def _update_unweighted_swa(self, l1_weights, l1_bias, l2_weights, l2_bias, l3_weights, l3_bias):
        """Update standard SWA running mean (average across tiles without performance weighting)"""
        for layer_name, layer_weights in [('l1', (l1_weights, l1_bias)), 
                                         ('l2', (l2_weights, l2_bias)), 
                                         ('l3', (l3_weights, l3_bias))]:
            weights, bias = layer_weights

            if len(weights.shape) == 3: 
                avg_weight = torch.mean(weights, dim=0)  # Simple average across tiles
            else:
                avg_weight = weights
            
            if self.swa_weights[layer_name]['weight'] is None:
                # First collection
                self.swa_weights[layer_name]['weight'] = avg_weight
                self.swa_weights[layer_name]['bias'] = bias
            else:
                # Update running average
                n = self.swa_weights[layer_name]['count']
                alpha = 1.0 / (n + 1)
                self.swa_weights[layer_name]['weight'] = (1.0 - alpha) * self.swa_weights[layer_name]['weight'] + alpha * avg_weight
                self.swa_weights[layer_name]['bias'] = (1.0 - alpha) * self.swa_weights[layer_name]['bias'] + alpha * bias
            
            self.swa_weights[layer_name]['count'] += 1
            
    def _update_weighted_swa(self, l1_weights, l1_bias, l2_weights, l2_bias, l3_weights, l3_bias, temperature=0.2):
        """Update performance-weighted SWA running mean"""
        # Get performance-based weights for each tile
        tile_weights = self.get_performance_weights(temperature)
        
        # Calculate weighted average across tiles for each layer
        l1_weighted_avg = torch.zeros_like(l1_weights[0])
        l2_weighted_avg = torch.zeros_like(l2_weights[0])
        l3_weighted_avg = torch.zeros_like(l3_weights[0])
        
        # Compute the weighted average across tiles
        for tile_idx, weight in enumerate(tile_weights):
            l1_weighted_avg += l1_weights[tile_idx] * weight
            l2_weighted_avg += l2_weights[tile_idx] * weight
            l3_weighted_avg += l3_weights[tile_idx] * weight
        
        # For biases (usually shared across tiles), just use the provided bias
        l1_bias_weighted = l1_bias
        l2_bias_weighted = l2_bias
        l3_bias_weighted = l3_bias
        
        # Update the running weighted averages
        for layer_name, layer_data in [('l1', (l1_weighted_avg, l1_bias_weighted)), 
                                      ('l2', (l2_weighted_avg, l2_bias_weighted)), 
                                      ('l3', (l3_weighted_avg, l3_bias_weighted))]:
            weighted_avg, bias = layer_data
            
            if self.weighted_swa_weights[layer_name]['weight'] is None:
                # First collection
                self.weighted_swa_weights[layer_name]['weight'] = weighted_avg
                self.weighted_swa_weights[layer_name]['bias'] = bias
            else:
                # Update running average
                n = self.weighted_swa_weights[layer_name]['count']
                alpha = 1.0 / (n + 1)
                self.weighted_swa_weights[layer_name]['weight'] = (1.0 - alpha) * self.weighted_swa_weights[layer_name]['weight'] + alpha * weighted_avg
                self.weighted_swa_weights[layer_name]['bias'] = (1.0 - alpha) * self.weighted_swa_weights[layer_name]['bias'] + alpha * bias
            
            self.weighted_swa_weights[layer_name]['count'] += 1
    
    def update_tile_loss(self, tile_idx, loss_value, momentum=0.9):
        """Update running loss for a specific tile."""
        if self.tile_losses[tile_idx] == 0.0 and self.loss_update_count[tile_idx] == 0:
            # First update
            self.tile_losses[tile_idx] = loss_value
        else:
            # Running average with momentum
            self.tile_losses[tile_idx] = momentum * self.tile_losses[tile_idx] + (1 - momentum) * loss_value
        
        self.loss_update_count[tile_idx] += 1
    
    def get_performance_weights(self, temperature=0.2):
        """
        Calculate weights based on tile performance (inverse loss).
        Higher temperature makes weights more uniform.
        """
        # Convert losses to inverse (better performance = higher weight)
        inv_losses = [1.0 / (loss + 1e-8) for loss in self.tile_losses]
        
        # Apply temperature scaling and softmax for more balanced weighting
        if temperature > 0:
            # Scale by temperature
            scaled_inv_losses = [inv_loss / temperature for inv_loss in inv_losses]
            
            # Softmax calculation
            max_val = max(scaled_inv_losses)
            exp_vals = [math.exp(val - max_val) for val in scaled_inv_losses]
            sum_exp = sum(exp_vals)
            weights = [exp_val / sum_exp for exp_val in exp_vals]
        else:
            # Just normalize
            sum_inv = sum(inv_losses)
            weights = [inv_loss / sum_inv for inv_loss in inv_losses]
            
        return weights
    
    def get_averaged_model(self, weighted=False):
        """Create a new model with SWA weights."""
        # Create new model with same config
        new_model = copy.deepcopy(self.model)
        new_model.eval()  # Put in eval mode
        
        # Choose which set of weights to use
        source_weights = self.weighted_swa_weights if weighted else self.swa_weights
        
        # Set up the layers with averaged weights
        new_model.l1_n = nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0], bias=True)
        new_model.l1_n.weight.data = source_weights['l1']['weight']
        new_model.l1_n.bias.data = source_weights['l1']['bias']
        
        new_model.l2_n = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], bias=True)
        new_model.l2_n.weight.data = source_weights['l2']['weight']
        new_model.l2_n.bias.data = source_weights['l2']['bias']
        
        new_model.l3_n = nn.Linear(HIDDEN_SIZES[1], OUTPUT_SIZE, bias=True)
        new_model.l3_n.weight.data = source_weights['l3']['weight']
        new_model.l3_n.bias.data = source_weights['l3']['bias']
        
        return new_model


def train(train_set, validation_dataset, model, optimizer, scheduler, classifier):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
    """
    time_init = time()
    l = []
    a = []
    num_tiles = model.l1.rpu_config.batch_size

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

            # Calculate inputs per tile as in the forward pass
            batch_size = images.shape[0]
            inputs_per_tile = (batch_size + num_tiles - 1) // num_tiles

            # Initialize list for tile losses
            tile_losses = []

            # # Calculate loss for each tile using the same distribution logic
            # for i in range(num_tiles):
            #     # Get the start and end indices for this tile
            #     start_idx = i * inputs_per_tile
            #     end_idx = min(start_idx + inputs_per_tile, batch_size)
                
            #     # Skip if this tile didn't receive any inputs
            #     if start_idx >= batch_size:
            #         continue
                    
            #     # Get outputs and targets for this tile
            #     tile_output = output[start_idx:end_idx]
            #     tile_target = labels[start_idx:end_idx]
                
            #     # Calculate loss
            #     tile_loss = classifier(tile_output, tile_target)
            #     tile_losses.append(tile_loss.item())
                
            #     # Update running loss average
            #     swa_tracker.update_tile_loss(i, tile_loss.item())

            # Run training (backward propagation).
            loss.backward()

            # Optimize weights.
            optimizer.step()

        # Evaluate the trained model.
        # model.digitize_model()
        # loss, acc = test_evaluation(model, validation_dataset, classifier)
        # print("Epoch {} - Test loss: {:.3f}".format(epoch_number, loss))
        # print("           Accuracy: {:.3f}\n".format(acc))
        # l.append(loss)
        # a.append(acc)

        # Update SWA tracker
        swa_tracker.update(epoch_number, collect_performance=False)


        # Decay learning rate if needed.
        #scheduler.step()

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

    num_tiles = [1, 4, 8, 16, 32]
    batch_size = 64

    # Load datasets.
    train_dataset, validation_dataset = load_images(batch_size)

    for nt in num_tiles:
        lr = 0.1
        # Run multiple times to average results.
        ma_acc = []
        swa_acc = []
        weighted_swa_acc = []
        for i in range(1):
            print(f'Run {i}')
            # Prepare the model.
            rpu_config = AGADEcRamPreset()
            rpu_config.device.unit_cell_devices = [EcRamPresetDevice(
                    gamma_up = -0.38,
                    gamma_down = 0.76,
                    allow_increasing=True
                ) for _ in range(2)]
            rpu_config.batch_size = nt
            rpu_config.device.fast_lr = lr * 1
            t = [1, 1, 1]
            model = Analog_Network(rpu_config=rpu_config, t=t, max_iter=20000, w_init=0.4, max_iter_zero=100000).to(DEVICE)
            swa_tracker = SWATracker(model, swa_start=35, swa_freq=1)

            classifier = nn.NLLLoss()
            optimizer = AnalogSGD(model.parameters(), lr=lr)
            optimizer.regroup_param_groups(model)
            scheduler = StepLR(optimizer, step_size=40, gamma=0.5)

            # Train the model.
            loss, acc = train(train_dataset, validation_dataset, model, optimizer, scheduler, classifier)
            model.digitize_model()
            ma_acc.append(test_evaluation(model, validation_dataset, classifier)[1])
            swa_model = swa_tracker.get_averaged_model(weighted=False)
            swa_acc.append(test_evaluation(swa_model, validation_dataset, classifier)[1])
            # weighted_swa_model = swa_tracker.get_averaged_model(weighted=True)
            # weighted_swa_acc.append(test_evaluation(weighted_swa_model, validation_dataset, classifier)[1])

        model_average_accuracy = np.mean(ma_acc)
        swa_accuracy = np.mean(swa_acc)
        # weighted_swa_accuracy = np.mean(weighted_swa_acc)
        print(f'Num tiles {nt}')
        print(f'    Mean Digital Accuracy: {model_average_accuracy}')
        print(f'    SWA Accuracy: {swa_accuracy}')
        # print(f'    Weighted SWA Accuracy: {weighted_swa_accuracy}')
        results.append({
            'num_tiles': nt,
            'model_average_accuracy': ma_acc,
            'swa_accuracy': swa_acc,
            # 'weighted_swa_accuracy': weighted_swa_acc
        })

    # save results to file
    df = pd.DataFrame(results)
    df.to_csv('SHD_script/mnist_80_v2.csv', index=False)