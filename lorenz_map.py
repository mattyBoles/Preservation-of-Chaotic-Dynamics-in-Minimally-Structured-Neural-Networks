import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path
from lorenz import LorenzGenerator
from scipy.optimize import curve_fit
from models import tanh_model
import json




def exp_model(z, a, b, c):
    return a * np.exp(b * z) + c

def softplus(x, b):
      return 1/b * (np.log(1 + np.exp(b*x)))

def plot_model_retrun_map(MODEL_NAME: str,
                          x0: np.ndarray,
                          n_transient: int = 5000,
                          n_steps:int = 1000):
    '''
    Plots the return map of a model, compared to that of the true system.
    This is a tent shaped function which indictaes teh next local maxima in the z direction, given the previous local maxima of z.
    We use this to show/analyse rarely visited areas of models, usually common in low width.

    Inputs:
        MODEL_NAME (str): The name of the model, as it appears in ".\output\".
        x0 (np.ndarray): The starting point, shape (3,).
        n_transient (int): Transient phase to ensure we are on the attrctor before taking z-maximas into consideration.
        n_steps (int): Number of steps to integrate over.
    '''
    
    x = x0
    c = LorenzGenerator()

    traj = c.generate_trajectory(x0=x, n_steps = (n_transient+n_steps), dt=0.01)

    z = traj[n_transient:,2]
    peak_indices, _ = find_peaks(z, distance=20)  # distance prevents false peaks
    z_maxima = z[peak_indices]
    zn, zn1 = z_maxima[:-1], z_maxima[1:]

    # sort by zn to find where the map peaks
    order = np.argsort(zn)
    zn_sorted, zn1_sorted = zn[order], zn1[order]
    peak_idx = np.argmax(zn1_sorted)   # location of the map's peak along x

    mask_lower = zn <= zn_sorted[peak_idx]   # rising branch (small z_n)
    mask_upper = zn >  zn_sorted[peak_idx]   # falling branch (large z_n)

    z_range_low  = np.linspace(zn[mask_lower].min(), zn[mask_lower].max(), 500)
    z_range_high = np.linspace(zn[mask_upper].min(), zn[mask_upper].max(), 500)

    #fit a 6th degree polynomial
    coeffs_left  = np.polyfit(zn[mask_lower],  zn1[mask_lower],  6)
    coeffs_right = np.polyfit(zn[mask_upper],  zn1[mask_upper],  6)

    func_left  = np.poly1d(coeffs_left)
    func_right = np.poly1d(coeffs_right)

    # then just call them like normal functions
    y_left  = func_left(z_range_low)
    y_right = func_right(z_range_high)


    #get model stats
    stats = torch.load(Path(r".\output", MODEL_NAME, MODEL_NAME + "_stats.pt"))
    mean, std = stats['mean'], stats['std']
    mean = mean.detach().numpy()
    std = std.detach().numpy()

    with open(Path(r'.\output', MODEL_NAME, MODEL_NAME+'_train.json'), 'r') as f:
                model_info = json.load(f)
                activation = model_info['ACTIVATION']
                hidden_size = model_info['HIDDEN_SIZE']
                beta = 1

    model = tanh_model(hidden_size, activation, beta)
    model.load_state_dict(torch.load(Path(r".\output", MODEL_NAME, MODEL_NAME + "_best_epoch.pth")))

    W1 = model.linear1.weight.detach().numpy()
    b1 = model.linear1.bias

    W2 = model.linear2.weight.detach().numpy()
    b2 = model.linear2.bias

    x_model_traj = []

    x_model = ((x0 - mean) / std)

    for _ in range(n_transient + n_steps):
        x_model = W1 @ x_model.reshape(-1,1) + b1.reshape(-1,1).detach().numpy()
        x_model = softplus(x_model, 1)
        
        x_model =W2@(x_model)+ b2.reshape(-1,1).detach().numpy()
        
        x_model_traj.append(((x_model[:,0]*std)+mean))

    z_model = np.asarray(x_model_traj)[n_transient:,2]
    peak_indices, _ = find_peaks(z_model, distance=20)  # distance prevents false peaks

    z_maxima = z_model[peak_indices]

    zn, zn1 = z_maxima[:-1], z_maxima[1:]

    fig, ax = plt.subplots()

    x_model_traj = np.asarray(x_model_traj)
    #ax.plot(x_model_traj[:,0],x_model_traj[:,1], x_model_traj[:,2])

    ax.scatter(z_maxima[:-1], z_maxima[1:])

    ax.plot(z_range_low,  y_left,  color='red',   linewidth=2)
    ax.plot(z_range_high, y_right, color='green', linewidth=2)
    ax.set_xlabel('Z_n')
    ax.set_ylabel('Z_n+1')
    ax.set_title('Return Map of Model, Compared with the True System')

    plt.show()
    plt.close('all')

    # plt.figure()
    # plt.plot(z_model)
    # plt.title("Model z trajectory")
    # plt.show()



if __name__ == '__main__':
      plot_model_retrun_map(MODEL_NAME = 'softplus_model', x0 = np.array([1,1,0]), n_steps=10000)