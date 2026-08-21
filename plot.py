import matplotlib.pyplot as plt
import numpy as np
import torch
from lorenz import LorenzGenerator
from pathlib import Path

def plot_model(model:torch.nn.Module,
               x0:np.ndarray,
               n_steps: int,
               mean: torch.Tensor,
               std: torch.Tensor,
               output_dir: str,
               MODEL_NAME: str) -> None:

    '''
    Plot an autoregressive trajectory of a model.

    Inputs:
        model (torch.nn.Module): The mdoel to generate a trajectory of.
        x0 (np.ndarray): The starting point, (1,3)
        n_steps (int): How many steps to plot.
        mean (torch.Tensor): The mean of x, y, z of the trian set, for z-score normalisation.
        std (torch.Tensor): The std of x, y, z of the trian set, for z-score normalisation.
        output_dir (str): Where the .pngs will end up.
        MODEL_NAME (str): The name of the directpry and model, where it will end up.
    '''
    
    model = model.to('cpu')

    generator = LorenzGenerator()
    model.eval()

    x_model = []
    x_model.append(x0)

    x = ((torch.tensor(x0) - mean) / std).float()
    
    with torch.inference_mode():
        for i in range(n_steps):
            x = model(x)
            x_model.append(np.array((x * std) + mean))

    generator.plot(png_name = f'{output_dir}/{MODEL_NAME}_MODEL_TRAJ.png', traj1=np.array(x_model), traj2 = None)



def plot_loss(trn_results: dict,
              output_dir: str) -> None:
    train_loss = trn_results['train_loss']
    val_loss = trn_results['val_loss']
    epochs = np.arange(1,len(train_loss)+1, 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, np.log(np.asarray(train_loss)), label = 'Train MSE')
    ax.plot(epochs, np.log(np.asarray(val_loss)), label = 'Val MSE')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.set_title('Train and Validation Loss over Epochs')
    ax.legend()

    output_path = Path(output_dir, Path(output_dir).parts[-1] + '_loss_curve.png')
    plt.savefig(output_path)