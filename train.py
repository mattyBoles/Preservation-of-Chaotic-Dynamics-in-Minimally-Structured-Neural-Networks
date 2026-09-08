import numpy as np
import torch
import pandas as pd
import random
import os
import json
from pathlib import Path

from data import traj_Dataset, lorenz_96_Dataset
from models import Lorenz_model, avg_euclidean_error, WeightedMSELoss
from engine import train, evaluate
from plot import plot_model, plot_loss, plot_96



def train_model(config:dict) -> tuple[str, float, float, float]:
    '''
    Trains a model from scratch. Also creates a best_model (based on lowest val loss) and last_model .pth to be loaded,
    stats.pt, which contain the mean and std of the train set for inferance, _MODEL_TRAJ, showing a typical trajectory,
    and a trin json, shwoing loss, Avg Euclidean distance, hyperparameters, and Lyapunov spectrum.

    Inputs:
        config (dict):
            'MODEL_NAME' (str): The name the model will be saved to.
            'N' (int): dimensions of L96,
            'F' (float): F in L96,
            'NUM_EPOCHS (int): The number of epochs to run for.
            'hidden_size' (int): The width of the hidden layer.
            'n_traj' (int): The numebr of different trajectories to train on.
            'traj_length' (int): The number of consecutive points on each trajectory to train on.
            'activation' (torch.nn.Module, Callable): The activation to train on, e.g. torch.nn.Softplus(beta=0.5).
            'beta' (float): The beta parameter in softplus.
            'random_seed' (int): The random seed to use for data generation.
            'mode' (int): 63 or 96, only necessary for datasets and plotting.

    Returns:
        dict: A dictionary containing:
            "MODEL_NAME", "NUM_EPOCHS", "NUM_TRAJ", "BETA", "TRAJ_LENGTH", "ACTIVATION", "HIDDEN_SIZE", "TRAIN_LOSS", "TRAIN_AVERAGE_EUCLIDEAN_DISTANCE", "VAL_LOSS", 
            "VAL_AVERAGE_EUCLIDEAN_DISTANCE", "TEST_LOSS", "TEST_AVERAGE_EUCLIDEAN_DISTANCE",

    '''
    device = 'cuda:0' if torch.cuda.is_available() == True else 'cpu'
    print(device)

    RANDOM_SEED = config['random_seed']

    #REPRODUCABILITY
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.use_deterministic_algorithms(True)

    MODEL_NAME = config['MODEL_NAME']

    output_dir = f'./betas'
    os.makedirs(output_dir, exist_ok=True)

    n_trajectories = config['n_traj']
    n_samples_per_traj = config['traj_length']
    n_transient = 5000
    dt = 0.01


    if config['mode'] == 63:
        dataset_generator = traj_Dataset
    elif config['mode'] == 96:
        dataset_generator = lorenz_96_Dataset
    else:
        raise ValueError("Invalid Mode Selected: Use 63 or 96")

    train_set = dataset_generator(n_trajectories=n_trajectories,
                                  n_samples_per_traj=n_samples_per_traj,
                                  n_transient=n_transient,
                                  dt=dt,
                                  mean = None,
                                  std = None,
                                  RANDOM_SEED = RANDOM_SEED)
    

    mean = train_set.mean
    std = train_set.std



    val_set = dataset_generator(n_trajectories=max(int(n_trajectories/8),4),
                                n_samples_per_traj=n_samples_per_traj,
                                n_transient=n_transient,
                                dt=dt,
                                mean = mean,
                                std = std,
                                RANDOM_SEED = RANDOM_SEED*10)
    
    test_set = dataset_generator(n_trajectories=max(int(n_trajectories/8),4),
                                 n_samples_per_traj=n_samples_per_traj,
                                 n_transient=n_transient,
                                 dt=dt,
                                 mean = mean,
                                 std = std,
                                 RANDOM_SEED = RANDOM_SEED*100)


    lr = 1.0
    NUM_EPOCHS = config['NUM_EPOCHS']


    train_loader = torch.utils.data.DataLoader(train_set, batch_size = len(train_set), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = len(val_set), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = len(test_set), shuffle=False)

    model = Lorenz_model(config['hidden_size'], config['activation'], input_size = config['N'], RANDOM_SEED=RANDOM_SEED).to(device)

    loss_fn = WeightedMSELoss(std=std)
    optimiser = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=10000,
        history_size=100,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe"
    )

    err_fn = avg_euclidean_error(mean = mean,
                                std = std)

    train_results = train(model = model,
                          train_loader = train_loader,
                          val_loader = val_loader,
                          loss_fn = loss_fn,
                          optimiser = optimiser,
                          err_fn = err_fn,
                          NUM_EPOCHS = NUM_EPOCHS,
                          device = device)
    

    best_val_loss = train_results['val_loss'].index(min(train_results['val_loss']))
    model.load_state_dict(train_results['model_statedict'][best_val_loss])
    torch.save(model.state_dict(), f'{output_dir}/{MODEL_NAME}_best_epoch.pth')


    trn_loss, trn_avg_err = evaluate(model = model,
                                 dataloader = train_loader,
                                 loss_fn = loss_fn,
                                 err_fn = err_fn,
                                 std=std,
                                 device = device)
    
    val_loss, val_avg_err = evaluate(model = model,
                                 dataloader = val_loader,
                                 loss_fn = loss_fn,
                                 err_fn = err_fn,
                                 std=std,
                                 device = device)
    
    test_loss, test_avg_err = evaluate(model = model,
                                   dataloader = test_loader,
                                   loss_fn = loss_fn,
                                   err_fn = err_fn,
                                   std=std,
                                   device = device)
    


    print('\n\n')
    print('-----RESULTS-----')
    print(f'| Train MSE : {trn_loss:.5f} | Train Average Euclidean Distance: {trn_avg_err:.5f} |\n')
    print(f'| Val MSE : {val_loss:.5f} | Val Average Euclidean Distance: {val_avg_err:.5f} |\n')
    print(f'| Test MSE : {test_loss:.5f} | Test Average Euclidean Distance: {test_avg_err:.5f} |\n')


    torch.save(
        {"mean": mean, "std": std},
        f'{output_dir}/{MODEL_NAME}_stats.pt'
    )

    def to_py_float(x, dp: int = 7) -> float:
        '''
        Converts a torch.Tensor / np.generic / plain number to a native
        Python float, rounded to `dp` decimal places, for JSON serialization.
        '''
        if hasattr(x, "item"):  # torch.Tensor, np.generic
            x = x.item()
        return round(float(x), dp)


    output_dict = {
        "MODEL_NAME":MODEL_NAME,
        "NUM_EPOCHS": config['NUM_EPOCHS'],
        "N": config["N"],
        "NUM_TRAJ": config['n_traj'],
        "TRAJ_LENGTH": config['traj_length'],
        "ACTIVATION": str(config['activation']),
        "HIDDEN_SIZE": config['hidden_size'],
        'BETA': config['beta'],
        "TRAIN_LOSS": to_py_float(trn_loss),
        "TRAIN_AVERAGE_EUCLIDEAN_DISTANCE": to_py_float(trn_avg_err),
        "VAL_LOSS" : to_py_float(val_loss),
        "VAL_AVERAGE_EUCLIDEAN_DISTANCE": to_py_float(val_avg_err),
        "TEST_LOSS" : to_py_float(test_loss),
        "TEST_AVERAGE_EUCLIDEAN_DISTANCE": to_py_float(test_avg_err)}

    with open(Path(output_dir, f"{MODEL_NAME}_train.json"), "w") as f:
        json.dump(output_dict, f, indent=2, default=str)
    


    if config["mode"] == 63:
        plot_model(model = model,
                   x0 = np.array([1,1,25]),
                   n_steps = 10000,
                   mean = mean,
                   std = std,
                   output_dir=output_dir,
                   MODEL_NAME=MODEL_NAME)

    elif config["mode"] == 96:
        plot_96(model = model,
                x0 = np.array([8.01, 8, 8, 8, 8, 8, 8, 8]),
                n_transient=5000,
                n_steps = 1000,
                mean = mean,
                std = std,
                output_dir=output_dir,
                MODEL_NAME=MODEL_NAME)
    else:
        raise ValueError("Invalid Mode Selected: Use 63 or 96")


    plot_loss(trn_results = train_results,
              output_dir=output_dir)
    

    return output_dict




if __name__ == '__main__':
    config = {
        "MODEL_NAME": 'lorenz96',
        'NUM_EPOCHS': 200,
        'N': 8,
        'F': 8,
        'hidden_size': 20,
        'n_traj': 200,
        'traj_length': 5,
        'activation': torch.nn.Softplus(),
        'beta': 1,
        'random_seed': 1}

    output = train_model(config=config)