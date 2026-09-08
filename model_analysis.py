import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time

from models import Lorenz_model
from lorenz_96 import lorenz_96
from pathlib import Path
from torch.func import jacrev



def analysis(model_info: dict,
             root_folder: str,
             mode: int,
             x0: np.ndarray,
             dt: float = 0.01,
             QR_steps: int = 10,
             transient_steps: int =1000,
             trajectory_steps: int = 10000) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Calculates singular values, lyapunov spectrum, mins of SVs and minimum gradients of SVs from a model of either type (L63 or L96). Lyapunov using Benitton method.
    We set Q = I. We disregard a transeint period to ensure we are on the attractor. Then, every step we evolve Q by the Jacobian (via RK4), and accumulate SVs.
    Every so many steps, we orthonormalise Q again, and accumulate diag(R), from QR decomp.

    Inputs:
        model_info (dict): A dictonary as outputeted by the train function, must include entries "MODEL_NAME", "HIDDEN_SIZE", "ACTIVATION" (a str which right now must contain 'softplus', 'tanh', or 'relu')
        root_folder (str): The path to the folder in which we find {MODEL_NAME}_best_epoch.pth and {MODEL_NAME}_stats.pt.
        mode (int): 63 or 96, refering to which dynamical system were looking at.
        x0 (np.ndarray): The starting coordinates, unnormalised.
        dt (float): timestep length
        QR_steps (int): Number of steps betweeen diag(R) accumulations and Q reorthonomalisations.
        transient_steps(int): The number of timesteps to through away at the start, to ensure we are on the attractor.
        trajectory_steps(int): The number of steps (excluding transient) to accumulate values over.
    
    Returns:
        Tuple[
            svs (np.ndarray): The singular values of the flow map, accumulated every step, in shape (timesteps, input_size),
            lyapunov_spectrum (np.ndarray): The long-term averaged rate of seperation of infinitesimal perterbations, of shape (input_size,)
            mins (np.ndarray): The global minima of the singular values, for comparison of local geometry. Shape (input_size,)
            min_grads (np.ndarray): The global minima of the gradients of the singular values, for comparison of local geometry. Via finite difference scheme delta(SV)/dt. Shape (input_size,)
            ]
    ''' 
    
    MODEL_NAME = model_info['MODEL_NAME']
    width = model_info['HIDDEN_SIZE']
    activation = model_info['ACTIVATION']

    try:
        beta = model_info['BETA']
    except:
        beta = None

    #Obviously not ideal but i don't know a better way to do this
    if 'softplus' in activation.lower():
        activation = torch.nn.Softplus(beta=beta)
    elif 'tanh' in activation.lower():
        activation = torch.nn.Tanh()
    elif 'relu' in activation.lower():
        activation =  torch.nn.ReLU()

    if mode == 96:
        input_size = model_info['N']
    elif mode == 63:
        input_size = 3

    model = Lorenz_model(hidden_units=width, activation=activation, input_size=input_size)
    model.load_state_dict(torch.load(Path(root_folder, f"{MODEL_NAME}_best_epoch.pth" )))

    mean, std = torch.load(Path(root_folder, f"{MODEL_NAME}_stats.pt"))['mean'], torch.load(Path(root_folder, f"{MODEL_NAME}_stats.pt"))['std']

    x = ((torch.tensor(x0) - mean) / std)

    svs = []
    lambda_ = np.empty((input_size,0))
    model.eval()
    jacobian_fn = jacrev(model)
    Q = np.eye(input_size)

    with torch.inference_mode():
        for i in range(transient_steps):
            x = x.float()
            x = model(x)
        for i in range(trajectory_steps):
            J = jacobian_fn(x.detach()).detach().numpy()
            J = np.squeeze(J)
            x = x.float()
            with torch.inference_mode():
                x = model(x)

            #Benittin
            Q = J @ Q
            J_physical = np.diag(std.detach().numpy()) @ J @ np.diag(1/std.detach().numpy())
            S= np.linalg.svd(J_physical, compute_uv=False)
            svs.append(S)

            if (i+1) % QR_steps == 0:
                                
                Q, R = np.linalg.qr(Q)
                lambda_ = np.hstack([
                lambda_,
                (np.log(np.abs(np.diag(R)) + 1e-10) / (dt * QR_steps))[:,None]
            ])

        lyapunov_spectrum = np.mean(lambda_, axis=1)
        svs = np.array(svs)

        #We calculate this to roughly look at the solution, whether it has a good smoothness or not. It's rough but it works.
        grads = (svs[1:,:] - svs[:-1,:])/dt
        min_grads = np.min(grads, axis=0)
        mins = np.min(svs, axis=0) 

        return svs, lyapunov_spectrum, mins, min_grads



if __name__ == '__main__':
    df = pd.read_csv(r".\results\lorenz63.csv")
    model_info = df.iloc[0]
    with open(r".\output\2026-07-11T19-51-00_relu_32\2026-07-11T19-51-00_relu_32_train.json", "r") as f:
        model_info = json.load(f)    
    x0 = np.array([
    np.random.uniform(-20, 20),
    np.random.uniform(-20, 20),
    np.random.uniform(0, 50)
    ])
    svs, lyapunov_spectrum, mins, min_grads = analysis(model_info=model_info,
                                      root_folder=r".\output\2026-07-11T19-51-00_relu_32",
                                      mode=63,
                                      x0=x0)

    c = lorenz_96(8,8,0.01)

    svs = c.find_svs(x=np.array([8.01,8,8,8,8,8,8,8]), trajectory_steps=10000)

    mins = np.min(svs, axis=0)
    grads = (svs[1:,:] - svs[:-1,:])/c.dt
    min_grads = np.min(grads, axis=0)

    print(mins.shape, min_grads.shape)


