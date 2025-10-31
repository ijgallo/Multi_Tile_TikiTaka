
import nnfs
import numpy as np
import torch
import torch.nn as nn
from nnfs.datasets import spiral_data
from aihwkit.simulator.presets import TikiTakaEcRamPreset
from aihwkit.optim import AnalogSGD
from training_functions import training_run, Analog_Network_spiral_trad
import pandas as pd

if __name__ == '__main__':
    # grid search for hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    nnfs.init()
    X_np, y_np = spiral_data(samples=1000, classes=3)
    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    y = torch.tensor(y_np, dtype=torch.int64).to(device)

    batch_size = [16, 32, 64, 128, 256]
    noise = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]

    results = []

    for bz in batch_size:
        lr = 0.0002 * bz
        for n in noise:
            l = []
            a = []
            for _ in range(5):
                rpu = TikiTakaEcRamPreset()
                rpu.device.fast_lr = lr
                t = [1, 1]
                rpu.forward.out_noise = n
                model = Analog_Network_spiral_trad(rpu_config=rpu, t=t, device=device)
                optimizer = AnalogSGD(model.parameters(), lr=lr)
                loss, accuracy = training_run(X, y, model, optimizer, nn.CrossEntropyLoss(), 1000, batch_size=bz)
                loss = np.mean(loss[-50:])
                accuracy = np.mean(accuracy[-50:])
                l.append(loss)
                a.append(accuracy)
            print(f'Loss: {loss:.2f}, accuracy: {accuracy:.2f} for batch size {bz} and noise {n}')
            results.append({
                'batch_size': bz,
                'noise': n,
                'losses': l,
                'accuracies': a,
            })

    # save results to file
    df = pd.DataFrame(results)
    df.to_csv('training_results_simple2.csv', index=False)


