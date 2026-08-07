import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

from models import tanh_model, parameterised_beta_model
from lorenz import LorenzGenerator
from pathlib import Path

def analysis(MODEL_NAME: str,
             dt: float = 0.01,
             QR_steps: int = 10,
             transient_steps: int = 5000,
             trajectory_steps: int = 10000,
             device: torch.device = 'cuda:0' if torch.cuda.is_available() else 'cpu') -> tuple[float, float, float, list[np.ndarray]]:
    '''
    This function analyses a models local geometry (via SVD) and long term seperation (via Lyapnuov spectra).
    We integrate over a number of transient sets.
    Every timestep we update Q with J.We find the physical J by doing std * J * 1/std.
    We find singular values and add them to a list.
    Every so often we renormalise Q and add thr diagonal of R to the list of lambdas to be averaged.

    Inputs:
        MODEL_NAME (str): Name of the model to analyse, as it appears in r".\\output\\"
        dt (float): Length of the timestep.
        QR_steps (int): Number of steps between each QR re-orthonormalisation and recording of lambda1,2,3

    '''
    
    root_folder = Path(r".\output", MODEL_NAME)
    mean, std = torch.load(Path(root_folder,MODEL_NAME+"_stats.pt"))['mean'].to(device), torch.load(Path(root_folder,MODEL_NAME+"_stats.pt"))['std'].to(device)
    with open(Path(root_folder, MODEL_NAME+'_train.json'), 'r') as f:
            model_info = json.load(f)
            activation = model_info['ACTIVATION']
            hidden_size = model_info['HIDDEN_SIZE']
            beta = 1

    model = tanh_model(hidden_units=hidden_size, activation=activation, beta=beta).to(device)
    model.load_state_dict(torch.load(Path(root_folder,MODEL_NAME+"_best_epoch.pth")))
    model = model.to(device)

    '''
    PLAN:
    1. Start at x0
    2. Run transient — iterate x = model(x) for n_transient steps, no accumulation
    3. Set Q = identity(3), log_growth = zeros(3)
    4. For each step:
    - Compute J = autograd_jacobian(model, xn)
    - Evolve Q = J @ Q
    - Step x = model(xn)
    - Every 10 steps:
        - QR decompose: Q, R = qr(Q)
        - Accumulate: log_growth += log(abs(diag(R)))
        - Count: n_renorm += 1
    5. Lyapunov = log_growth / (n_renorm * 10 * dt)
    '''

    Q = np.eye(3)
    lambda_ = np.empty((3,0))
    x = torch.tensor([[1,1,0]], dtype=torch.float32)
    x = (x - mean)/std
    singular_values, l_vectors, r_vectors, zdot = [], [], [], []
    sigma = std.numpy()

    Q = np.eye(3)
    lambda_ = np.empty((3,0))
    x = torch.tensor([[np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(0,50)]])
    #x = torch.tensor([[2,4,9]])
    x = (x - mean)/std
    for i in range(transient_steps):
        x = x.float()
        x = model(x)

    for i in range(trajectory_steps):
        J = torch.autograd.functional.jacobian(model, x)
        J = J.squeeze().detach().numpy()

        x = x.float()
        x_ = model(x)
        zdot.append(((x_[0,2] - x[0,2]) / dt).detach())
        Q = J @ Q
        J_physical = np.diag(sigma) @ J @ np.diag(1/sigma)
        U, S, Vt = np.linalg.svd(J_physical)

        singular_values.append(S)
        l_vectors.append(U)
        r_vectors.append(Vt)

        if (i+1) % QR_steps == 0:
                    
            Q, R = np.linalg.qr(Q)
            lambda_ = np.hstack([lambda_, np.array([[np.log(np.abs(R[0,0])+1e-10)/(dt*QR_steps)],
                        [np.log(np.abs(R[1,1])+1e-10)/(dt*QR_steps)],
                        [np.log(np.abs(R[2,2])+1e-10)/(dt*QR_steps)]])])

        x = x_#for zdot

    Lyapunov_spectrum = np.mean(lambda_, axis=1)

    singular_values = np.array(singular_values)
    zdot = np.array(zdot)

    return Lyapunov_spectrum[0], Lyapunov_spectrum[1],Lyapunov_spectrum[2], singular_values

if __name__ == '__main__':
    # l1, l2, l3 = [],[],[]
    # for _ in range(20):
    lyapunov_11, lyapunov_21, lyapunov_31, sv1 = analysis(MODEL_NAME="softplus_model")
        # l1.append(lyapunov_11)
        # l2.append(lyapunov_21)
        # l3.append(lyapunov_31)

    # l1 = np.mean(np.asarray(l1))
    # l2 = np.mean(np.asarray(l2))
    # l3 = np.mean(np.asarray(l3))

    c = LorenzGenerator()
    results = c.find_lyapunov_spectrum(x=np.array([2,4,9]))

    sv_real = np.asarray(results['singular_values'])

    # grads_model = (sv1[1:,2] - sv1[:-1,2])/0.01
    # print('MODEL:')
    # print(np.min(grads_model))

    # grads_real = (sv_real[1:,2] - sv_real[:-1,2])/0.01
    # print('REAL:')
    # print(np.min(grads_real))


    print(f"REal: {np.min(sv_real[:,2])}")
    print(f"REal: {np.max(sv_real[:,2])}")

    #print(f"Model: {np.min(sv1[:,2])}")    

    #print(f"Lyapunov1: {lyapunov_11}\nLyapunov2: {lyapunov_21}\nLyapunov3: {lyapunov_31}\n")
    fig, axes = plt.subplots(3,1)

    #axes[0].plot(sv1[:1000,0], label = 'SV1 - model')
    #axes[1].plot(sv1[:1000,1], label = 'SV2 - model')
    #axes[2].plot(sv1[:1000,2], label = 'SV3 - model')


    axes[0].plot(sv_real[:1000,0], label = 'SV1 - true')
    axes[1].plot(sv_real[:1000,1], label = 'SV2 -  true')
    axes[2].plot(sv_real[:1000,2], label = 'SV3 - true')

    axes[1].legend()
    axes[0].legend()
    axes[2].legend()
    fig.tight_layout()
    plt.show()
    print('done')

