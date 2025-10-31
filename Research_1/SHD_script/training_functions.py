import numpy as np
import torch
import torch
import torch.nn as nn
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets import TikiTakaEcRamPreset, TTv2EcRamPreset, ChoppedTTv2EcRamPreset, AGADEcRamPreset
from aihwkit.simulator.configs import PrePostProcessingParameter, InputRangeParameter
#from aihwkit.simulator.tiles.transfer_for_batched_TT import TorchTransferTile
from aihwkit.simulator.tiles.transfer_for_batched_TTv2_v2 import TorchTransferTile



def get_As_and_Cs(model):
    Cs, As = [], []
    for param in model.parameters():
        analog_tile = param.analog_tile
        # Access the TransferCompound device
        device_weights = analog_tile.get_hidden_parameters()
        A = device_weights['hidden_weights_0']
        C = device_weights['hidden_weights_1']
        As.append(A)
        Cs.append(C)
    return torch.stack(Cs), torch.stack(As)


def training_run(X, y, model, optimizer, loss_fn, epochs, batch_size=1):
    for _ in range(epochs):
        model.train()
        # reshuffle the data
        perm = torch.randperm(len(X))
        X = X[perm]
        y = y[perm]
        # batch the data
        X_batched = torch.split(X, batch_size)
        y_batched = torch.split(y, batch_size)
        for x, y_true in zip(X_batched, y_batched):
            output = model(x)
            l = loss_fn(output, y_true)
            # backward pass
            l.backward()
            # update params
            optimizer.step()
            optimizer.zero_grad()

    return model

def evaluate_model_digital(X, y, model, loss_fn):
    model.eval()
    real_output = model(X)
    loss = loss_fn(real_output, y).item()
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        correct_predictions = torch.sum(torch.argmax(real_output, axis=1) == y)
        accuracy = correct_predictions.item() / len(y)
    return loss, accuracy


def evaluate_model_analog(X, y, model, loss_fn):
    loss = []
    accuracy = []
    for _ in range(100):
        model.analog_eval()
        real_output = model(X)
        loss.append(loss_fn(real_output, y).item())
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            correct_predictions = torch.sum(torch.argmax(real_output, axis=1) == y)
            accuracy.append(correct_predictions.item() / len(y))
    return np.mean(loss), np.mean(accuracy)


class Analog_Network_spiral(nn.Module):
    
    def __init__(self, rpu_config=AGADEcRamPreset(), t=[5, 1], device='cpu'):
        super(Analog_Network_spiral, self).__init__()

        rpu_config.device.units_in_mbatch = True
        rpu_config.device.transfer_every = t[0]
        self.l1 = TorchTransferTile(2, 64, bias=True, rpu_config=rpu_config).to(device)
        rpu_config.device.transfer_every = t[1]
        self.l2 = TorchTransferTile(64, 3, bias=True, rpu_config=rpu_config).to(device)

        batch_size = self.l1.rpu_config.batch_size

        # initialize weights  
        l1_weights = self.l1.get_weights()[0][0]
        #innit_w = [torch.nn.init.xavier_normal_(l1_weights) for _ in range(rpu_config.batch_size)]
        innit_w = torch.nn.init.xavier_normal_(l1_weights)
        params = self.l1.get_hidden_parameters()
        params.update({f'slow_weight_{i+1}': innit_w for i in range(batch_size)})
        params.update({f'fast_weight': torch.zeros_like(innit_w)})
        self.l1.set_hidden_parameters(params)

        l2_weights = self.l2.get_weights()[0][0]
        innit_w = torch.nn.init.xavier_normal_(l2_weights)
        params = self.l2.get_hidden_parameters()
        params.update({f'slow_weight_{i+1}': innit_w for i in range(batch_size)})
        params.update({f'fast_weight': torch.zeros_like(innit_w)})
        self.l2.set_hidden_parameters(params) 

    def digital_eval(self):
        self.digital_eval = True
        self.training = False

    def analog_eval(self):
        self.digital_eval = False
        self.training = False   

    def forward(self, x):
        if self.training:
            x = torch.relu(self.l1(x)) 
            x = self.l2(x)
            return x * 25
        else:
            with torch.no_grad():
                w_l1 = torch.mean(self.l1.get_weights()[0], dim=0)
                w_l2 = torch.mean(self.l2.get_weights()[0], dim=0)
                if  self.digital_eval:
                    self.l1_n = nn.Linear(2, 64, bias=True).to(x.device)
                    self.l1_n.weight.data = w_l1
                    self.l1_n.bias.data = self.l1.get_weights()[1]
                    self.l2_n = nn.Linear(64, 3, bias=True).to(x.device)
                    self.l2_n.weight.data = w_l2
                    self.l2_n.bias.data = self.l2.get_weights()[1]
                else:
                    self.l1_n = AnalogLinear(2, 64, bias=True, rpu_config=self.l1.rpu_config).to(x.device)
                    self.l1_n.set_weights(w_l1, bias=self.l1.get_weights()[1])
                    w_l2 = torch.mean(torch.stack(self.l2.get_weights()[0]), dim=0)
                    self.l2_n = AnalogLinear(64, 3, bias=True, rpu_config=self.l2.rpu_config).to(x.device)
                    self.l2_n.set_weights(w_l2, bias=self.l2.get_weights()[1])

                x = torch.relu(self.l1_n(x))
                x = self.l2_n(x)
                return x * 25



class Analog_Network_spiral_v2(nn.Module):
    
    def __init__(self, rpu_config=AGADEcRamPreset(), t=[5, 1], device='cpu'):
        super(Analog_Network_spiral_v2, self).__init__()

        rpu_config.device.units_in_mbatch = True
        rpu_config.device.transfer_every = t[0]
        self.l1 = TorchTransferTile(2, 64, bias=True, rpu_config=rpu_config).to(device)
        rpu_config.device.transfer_every = t[1]
        self.l2 = TorchTransferTile(64, 3, bias=True, rpu_config=rpu_config).to(device)

        batch_size = self.l1.rpu_config.batch_size

        # initialize weights  
        l1_weights = self.l1.get_weights()[0][0]
        #innit_w = [torch.nn.init.xavier_normal_(l1_weights) for _ in range(rpu_config.batch_size)]
        innit_w = torch.nn.init.xavier_normal_(l1_weights)
        params = self.l1.get_hidden_parameters()
        params.update({f'slow_weight_{i+1}': innit_w for i in range(batch_size)})
        params.update({f'fast_weight_{i+1}': torch.zeros_like(innit_w) for i in range(batch_size)})
        self.l1.set_hidden_parameters(params)

        l2_weights = self.l2.get_weights()[0][0]
        innit_w = torch.nn.init.xavier_normal_(l2_weights)
        params = self.l2.get_hidden_parameters()
        params.update({f'slow_weight_{i+1}': innit_w for i in range(batch_size)})
        params.update({f'fast_weight_{i+1}': torch.zeros_like(innit_w) for i in range(batch_size)})
        self.l2.set_hidden_parameters(params) 

    def digital_eval(self):
        self.digital_eval = True
        self.training = False

    def analog_eval(self):
        self.digital_eval = False
        self.training = False   

    def forward(self, x):
        if self.training:
            x = torch.relu(self.l1(x)) 
            x = self.l2(x)
            return x * 25
        else:
            with torch.no_grad():
                w_l1 = torch.mean(self.l1.get_weights()[0], dim=0)
                w_l2 = torch.mean(self.l2.get_weights()[0], dim=0)
                if  self.digital_eval:
                    self.l1_n = nn.Linear(2, 64, bias=True).to(x.device)
                    self.l1_n.weight.data = w_l1
                    self.l1_n.bias.data = self.l1.get_weights()[1]
                    self.l2_n = nn.Linear(64, 3, bias=True).to(x.device)
                    self.l2_n.weight.data = w_l2
                    self.l2_n.bias.data = self.l2.get_weights()[1]
                else:
                    self.l1_n = AnalogLinear(2, 64, bias=True, rpu_config=self.l1.rpu_config).to(x.device)
                    self.l1_n.set_weights(w_l1, bias=self.l1.get_weights()[1])
                    self.l2_n = AnalogLinear(64, 3, bias=True, rpu_config=self.l2.rpu_config).to(x.device)
                    self.l2_n.set_weights(w_l2, bias=self.l2.get_weights()[1])

                x = torch.relu(self.l1_n(x))
                x = self.l2_n(x)
                return x * 25
            


class Analog_Network_spiral_trad(nn.Module):
    
    def __init__(self, rpu_config=TikiTakaEcRamPreset(), t=[5, 1], device='cpu'):
        super(Analog_Network_spiral_trad, self).__init__()
    
        rpu_config.device.transfer_every = t[0]
        self.l1 = AnalogLinear(2, 64, bias=True, rpu_config=rpu_config).to(device)
        rpu_config.device.transfer_every = t[1]
        self.l2 = AnalogLinear(64, 3, bias=True, rpu_config=rpu_config).to(device)

        # initialize weights using xavier
        self.l1.set_weights(weight = torch.nn.init.xavier_normal_(self.l1.get_weights()[0]), bias = torch.zeros_like(self.l1.get_weights()[1]))
        self.l2.set_weights(weight = torch.nn.init.xavier_normal_(self.l2.get_weights()[0]), bias = torch.zeros_like(self.l2.get_weights()[1])) #torch.randn_like(self.l2.get_weights()[0]) * 0.05
        
    def forward(self, x):
        x = torch.relu(self.l1(x)) 
        x = self.l2(x)
        return x * 25
    


