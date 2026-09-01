import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models import tanh_model
from lorenz_96 import lorenz_96
from pathlib import Path


def analysis(model_info: dict,
             root_folder: str,
             mode: int,
             x0: np.ndarray,
             dt: float = 0.01,
             QR_steps: int = 10,
             transient_steps: int = 5000,
             trajectory_steps: int = 10000):
    MODEL_NAME = model_info['MODEL_NAME']
    width = model_info['HIDDEN_SIZE']
    activation = model_info['ACTIVATION']
    beta = model_info['BETA']
    if 'softplus' in activation.lower():
        activation = torch.nn.Softplus(beta=beta)
    elif 'tanh' in activation.lower():
        activation = torch.nn.Tanh()
    if mode == 96:
        input_size = model_info['N']
    elif mode == 63:
        input_size = 3
    model = tanh_model(hidden_units=width, activation=activation, input_size=input_size)
    model.load_state_dict(torch.load(Path(root_folder, f"{MODEL_NAME}_best_epoch.pth" )))

    mean, std = torch.load(Path(root_folder, f"{MODEL_NAME}_stats.pt"))['mean'], torch.load(Path(root_folder, f"{MODEL_NAME}_stats.pt"))['std']

    x = ((torch.tensor(x0)-mean)/std)

    svs = []
    lambda_ = np.empty((input_size,0))

    Q = np.eye(input_size)
    for i in range(transient_steps):
        x=x.float()
        x = model(x)

    for i in range(trajectory_steps):
        J = torch.autograd.functional.jacobian(model, x)
        J = J.squeeze().detach().numpy()
        x = x.float()
        x_ = model(x)
        Q = J @ Q
        J_physical = np.diag(std.detach().numpy()) @ J @ np.diag(1/std.detach().numpy())
        U, S, Vt = np.linalg.svd(J_physical)
        svs.append(S)

        if (i+1) % 10 == 0:
                            
            Q, R = np.linalg.qr(Q)
            lambda_ = np.hstack([
            lambda_,
            (np.log(np.abs(np.diag(R)) + 1e-10) / (dt * QR_steps))[:,None]
        ])
        
        x = x_#for zdot

    lyapunov_spectrum = np.mean(lambda_, axis=1)
    svs = np.array(svs)

    grads = (svs[1:,:] - svs[:-1,:])/dt
    min_grads = np.min(grads, axis=0)
    mins = np.min(svs, axis=0)

    return svs, lyapunov_spectrum, mins, min_grads



if __name__ == '__main__':
    df = pd.read_csv(r".\results\lorenz63.csv")
    model_info = df.iloc[0]
    x0 = np.array([1,1,1])
    svs, lyapunov_spectrum, mins, min_grads = analysis(model_info=model_info,
                                      root_folder=r".\results\lorenz63",
                                      mode=63,
                                      x0=x0)

