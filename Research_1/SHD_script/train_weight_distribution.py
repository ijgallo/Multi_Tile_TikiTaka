import nnfs
import numpy as np
import torch
import torch.nn as nn
from nnfs.datasets import spiral_data
from aihwkit.simulator.presets import TikiTakaEcRamPreset
from aihwkit.optim import AnalogSGD
from training_functions import training_run, Analog_Network_spiral, evaluate_model_digital, evaluate_model_analog
import pandas as pd

if __name__ == '__main__':
    # grid search for hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    nnfs.init()
    X_np, y_np = spiral_data(samples=1000, classes=3)
    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    y = torch.tensor(y_np, dtype=torch.int64).to(device)

    num_tiles = [1, 4, 16, 64]
    batch_size = 64
    noise = [0.0]

    results = []

    for nt in num_tiles:
        print(f'Running for {nt} tiles')
        lr = 0.0001 * batch_size
        for n in noise:
            w1, b1, w2, b2 = [], [], [], []
            for i in range(5):
                rpu = TikiTakaEcRamPreset()
                rpu.batch_size = nt
                rpu.device.fast_lr = lr
                t = [1, 1]
                rpu.forward.out_noise = n
                model = Analog_Network_spiral(rpu_config=rpu, t=t, device=device)
                optimizer = AnalogSGD(model.parameters(), lr=lr)
                model = training_run(X, y, model, optimizer, nn.CrossEntropyLoss(), 1000, batch_size=batch_size)
                w1.append(torch.mean(torch.stack(model.l1.get_weights()[0]), dim=0).numpy().tolist())
                b1.append(model.l1.get_weights()[1].numpy().tolist())
                w2.append(torch.mean(torch.stack(model.l2.get_weights()[0]), dim=0).numpy().tolist())
                b2.append(model.l2.get_weights()[1].numpy().tolist())
                
            results.append({
                'num_tiles': nt,
                'noise': n,
                'w1': w1,
                'b1': b1,
                'w2': w2,
                'b2': b2
            })

    # save results to file
    df = pd.DataFrame(results)
    df.to_csv('training_weight_distribution_1000_noiseless.csv', index=False)