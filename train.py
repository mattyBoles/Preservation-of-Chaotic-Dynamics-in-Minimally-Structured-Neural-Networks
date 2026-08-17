import numpy as np
import torch
import pandas as pd
import random
import os
import json
from datetime import datetime, timezone
from pathlib import Path

from data import traj_Dataset
from models import tanh_model, avg_euclidean_error, parameterised_beta_model, WeightedMSELoss
from engine import train, test
from plot import plot_model, plot_loss
from model_analysis import analysis
from li_and_ravela import W1, b1, W2, b2


def train_model(config:dict) -> tuple[str, float, float, float]:
    '''
    Trains a model from scratch and does some quick lyapunov analysis on it. Also creates a best_model (based on lowest val loss) and last_model .pth to be loaded,
    stats.pt, which contain the mean and std of the train set for inferance, _MODEL_TRAJ, showing a typical trajectory,
    and a trin json, shwoing loss, Avg EUclidean distance, hyperparameters, and Lyapunov spectrum.

    Inputs:
        config (dict):
            'MODEL_NAME' (str): The name the model will be saved to.
            'NUM_EPOCHS (int): The number of epochs to run for.
            'hidden_size' (int): The width of the hidden layer.
            'n_traj' (int): The numebr of different trajectories to train on.
            'traj_length' (int): The number of consecutive points on each trajectory to train on.
            'activation' (str): The activation to train on, namely 'relu', 'tanh', 'arctan', or 'softplus'.
            'beta' (float): The beta parameter in softplus.
            'random_seed' (int): The random seed to use for data geenration.

    Returns:
        dict: A dictionary containing:
            "MODEL_NAME", "NUM_EPOCHS", "NUM_TRAJ", "TRAJ_LENGTH", "ACTIVATION", "HIDDEN_SIZE", "TRAIN_LOSS", "TRAIN_AVERAGE_EUCLIDEAN_DISTANCE", "VAL_LOSS", 
            "VAL_AVERAGE_EUCLIDEAN_DISTANCE", "TEST_LOSS", "TEST_AVERAGE_EUCLIDEAN_DISTANCE", "Lyapunov1", "Lyapunov2", "Lyapunov3",

    '''
    device = 'cuda:0' if torch.cuda.is_available() == True else 'cpu'
    print(device)

    RANDOM_SEED = config['random_seed']

    #REPRODUCABILITY
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.use_deterministic_algorithms(True)

    #MODEL_NAME = str(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")) + '_6_width_model'
    MODEL_NAME = config['MODEL_NAME']

    output_dir = f'./betas/'
    os.makedirs(output_dir, exist_ok=True)

    n_trajectories = config['n_traj']
    n_samples_per_traj = config['traj_length']
    n_transient = 5000
    dt = 0.01


    train_set = traj_Dataset(n_trajectories=n_trajectories,
                            n_samples_per_traj=n_samples_per_traj,
                            n_transient=n_transient,
                            dt=dt,
                            mean = None,
                            std = None,
                            RANDOM_SEED = RANDOM_SEED)
    

    mean = train_set.mean
    std = train_set.std



    val_set = traj_Dataset(n_trajectories=max(int(n_trajectories/8),4),
                            n_samples_per_traj=n_samples_per_traj,
                            n_transient=n_transient,
                            dt=dt,
                            mean = mean,
                            std = std,
                            RANDOM_SEED=RANDOM_SEED*10)
    
    test_set = traj_Dataset(n_trajectories=max(int(n_trajectories/8),4),
                            n_samples_per_traj=n_samples_per_traj,
                            n_transient=n_transient,
                            dt=dt,
                            mean = mean,
                            std = std,
                            RANDOM_SEED=RANDOM_SEED*100)


    #BATCH_SIZE = 64 for lbfgs, use full set
    lr = 1.0
    NUM_EPOCHS = config['NUM_EPOCHS']


    train_loader = torch.utils.data.DataLoader(train_set, batch_size = len(train_set), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = len(val_set), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = len(test_set), shuffle=False)

    model = tanh_model(config['hidden_size'], config['activation'], RANDOM_SEED=RANDOM_SEED).to(device)

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
    

    #torch.save(model.state_dict(), f'{output_dir}/{MODEL_NAME}_last_epoch.pth')

    best_val_loss = train_results['val_loss'].index(min(train_results['val_loss']))
    model.load_state_dict(train_results['model_statedict'][best_val_loss])
    torch.save(model.state_dict(), f'{output_dir}/{MODEL_NAME}_best_epoch.pth')


    trn_loss, trn_avg_err = test(model = model,
                            dataloader = train_loader,
                            loss_fn = loss_fn,
                            err_fn = err_fn,
                            std=std,
                            device = device)
    val_loss, val_avg_err = test(model = model,
                            dataloader = val_loader,
                            loss_fn = loss_fn,
                            err_fn = err_fn,
                            std=std,
                            device = device)
    test_loss, test_avg_err = test(model = model,
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
        "NUM_TRAJ": config['n_traj'],
        "TRAJ_LENGTH": config['traj_length'],
        "ACTIVATION": config['activation'],
        "HIDDEN_SIZE": config['hidden_size'],
        'BETA': config['beta'],
        "TRAIN_LOSS": to_py_float(trn_loss),
        "TRAIN_AVERAGE_EUCLIDEAN_DISTANCE": to_py_float(trn_avg_err),
        "VAL_LOSS" : to_py_float(val_loss),
        "VAL_AVERAGE_EUCLIDEAN_DISTANCE": to_py_float(val_avg_err),
        "TEST_LOSS" : to_py_float(test_loss),
        "TEST_AVERAGE_EUCLIDEAN_DISTANCE": to_py_float(test_avg_err)}

    # with open(Path(output_dir, f"{MODEL_NAME}_train.json"), "w") as f:
    #     json.dump(output_dict, f, indent=2, default=str)
    

    # ly1, ly2, ly3 = [],[],[]
    # for _ in range(20):
    #     l1, l2, l3, _ = analysis(MODEL_NAME=MODEL_NAME)
    #     ly1.append(l1)
    #     ly2.append(l2)
    #     ly3.append(l3)
    
    # l1 = np.mean(np.asarray(ly1))
    # l2 = np.mean(np.asarray(ly2))
    # l3 = np.mean(np.asarray(ly3))


    # output_dict.update({
    #         "Lyapunov1": l1,
    #         "Lyapunov2": l2,
    #         "Lyapunov3": l3,
    #     })

    # with open(Path(output_dir, f"{MODEL_NAME}_train.json"), "w") as f:
    #         json.dump(output_dict, f, indent=2, default=str)


    # plot_model(model = model,
    #         x0 = np.array([1,1,25]),
    #         n_steps = 10000,
    #         mean = mean,
    #         std = std,
    #         output_dir=output_dir,
    #         MODEL_NAME=MODEL_NAME)

    # plot_loss(trn_results = train_results,
    #           output_dir=output_dir)
    

    return output_dict




if __name__ == '__main__':
    config = {
        "MODEL_NAME": 'tanh',
        'NUM_EPOCHS': 200,
        'hidden_size': 32,
        'n_traj': 100,
        'traj_length': 5,
        'activation': torch.nn.Softplus,
        'beta': 1,
        'random_seed': random.randint(1,100)}

    output = train_model(config=config)