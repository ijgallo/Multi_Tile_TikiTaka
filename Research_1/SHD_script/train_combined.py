
import nnfs
import numpy as np
import torch
import torch.nn as nn
from nnfs.datasets import spiral_data
from aihwkit.simulator.presets import TikiTakaEcRamPreset, TTv2EcRamPreset, ChoppedTTv2EcRamPreset, AGADEcRamPreset
from aihwkit.optim import AnalogSGD
from training_functions import training_run, Analog_Network_spiral, Analog_Network_spiral_v2, evaluate_model_digital, evaluate_model_analog
import pandas as pd

if __name__ == '__main__':
    # grid search for hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    nnfs.init()
    X_np, y_np = spiral_data(samples=1000, classes=3)
    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    y = torch.tensor(y_np, dtype=torch.int64).to(device)

    num_tiles = [1, 4, 8, 16, 32, 64]
    batch_size = 64
    noise = [0.1]
    #noise = [0.0, 0.04, 0.08, 0.12, 0.16, 0.2]
    #noise = [0.075, 0.1, 0.125, 0.15, 0.175, 0.2]


    results = []

    for nt in num_tiles:
        lr = 0.0001 * batch_size
        for n in noise:
            digital_l, digital_a = [], []
            #analog_l, analog_a = [], []
            for i in range(10):
                #rpu = TikiTakaEcRamPreset()
                rpu = AGADEcRamPreset()
                rpu.device.in_chop_prob = 0.001
                rpu.batch_size = nt
                rpu.device.fast_lr = lr * 16
                t = [1, 1]
                rpu.forward.out_noise = n
                rpu.device.transfer_forward.out_noise = n
                model = Analog_Network_spiral_v2(rpu_config=rpu, t=t, device=device)
                optimizer = AnalogSGD(model.parameters(), lr=lr)
                model = training_run(X, y, model, optimizer, nn.CrossEntropyLoss(), 1000, batch_size=batch_size)
                dloss, daccuracy = evaluate_model_digital(X, y, model, nn.CrossEntropyLoss())
                #aloss, aaccuracy = evaluate_model_analog(X, y, model, nn.CrossEntropyLoss())
                digital_l.append(dloss), digital_a.append(daccuracy)#, analog_l.append(aloss), analog_a.append(aaccuracy)
                print(f'Run {i}:\n  Digital Accuracy: {daccuracy:.2f}')#\n  Analog Accuracy: {aaccuracy:.2f}
            print(f'Mean Digital Accuracy: {np.mean(digital_a):.2f}')#, Mean Analog Accuracy: {np.mean(analog_a):.2f} for num tiles {nt} and noise {n}
            results.append({
                'num_tiles': nt,
                #'noise': n,
                'digital_losses': digital_l,
                'digital_accuracies': digital_a,
                # 'analog_losses': analog_l,
                # 'analog_accuracies': analog_a,
            })

    # save results to file
    df = pd.DataFrame(results)
    df.to_csv('agad_v2_new.csv', index=False)