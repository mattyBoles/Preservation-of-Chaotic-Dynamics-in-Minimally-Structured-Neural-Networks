import matplotlib.pyplot as plt
import numpy as np
import torch
from lorenz_63 import lorenz_63
from pathlib import Path
import seaborn as sns
import pandas as pd

def plot_model(model:torch.nn.Module,
               x0:np.ndarray,
               n_steps: int,
               mean: torch.Tensor,
               std: torch.Tensor,
               output_dir: str,
               MODEL_NAME: str) -> None:

    '''
    Plot an autoregressive trajectory of a model (L63).

    Inputs:
        model (torch.nn.Module): The mdoel to generate a trajectory of.
        x0 (np.ndarray): The starting point, (3,)
        n_steps (int): How many steps to plot.
        mean (torch.Tensor): The mean of x, y, z of the trian set, for z-score normalisation.
        std (torch.Tensor): The std of x, y, z of the trian set, for z-score normalisation.
        output_dir (str): Where the .pngs will end up.
        MODEL_NAME (str): The name of the directpry and model, where it will end up.
    '''
    
    model = model.to('cpu')

    generator = lorenz_63()
    model.eval()

    x_model = []
    x_model.append(x0)

    x = ((torch.tensor(x0) - mean) / std).float().unsqueeze(0)
    
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
    plt.close('all')


def plot_96(model,
            x0: np.ndarray,
            n_steps: int,
            mean: torch.Tensor,
            std: torch.Tensor,
            output_dir: str,
            MODEL_NAME: str) -> None:
    '''
    Plot an autoregressive trajectory of a model (L96).

    Inputs:
        model (torch.nn.Module): The mdoel to generate a trajectory of.
        x0 (np.ndarray): The starting point, (N,)
        n_steps (int): How many steps to plot.
        mean (torch.Tensor): The mean of x, y, z of the trian set, for z-score normalisation.
        std (torch.Tensor): The std of x, y, z of the trian set, for z-score normalisation.
        output_dir (str): Where the .pngs will end up.
        MODEL_NAME (str): The name of the directpry and model, where it will end up.
    '''

    model_traj = []
    x_model = ((torch.tensor(x0) - mean)/std).float().unsqueeze(0)

    for _ in range(n_steps):
        x_model = x_model.float()
        x_model = model(x_model)
        model_traj.append((x_model * std + mean).detach().numpy())
        traj = np.asarray(model_traj)
        
    fig, ax = plt.subplots(figsize=(10, 3))

    im = ax.imshow(
        traj[:1000].T,
        aspect='auto',
        origin='lower',
        cmap='RdBu_r',
        interpolation='nearest'
    )

    ax.set_xlabel('Time step')
    ax.set_ylabel('Variable $i$')
    ax.set_yticks(range(8))
    ax.set_yticklabels(range(1, 9))
    ax.set_title(MODEL_NAME)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('$x_i$')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{MODEL_NAME}_traj.png')
    plt.close('all')


def plot_mse_min_grad(df: pd.DataFrame,
                      width: int,
                      activation: str) -> None:

    df = df[df['HIDDEN_SIZE'] == width]
    df = df[df['ACTIVATION'] == activation]

    fig, ax = plt.subplots(figsize=(10,10))
    sns.scatterplot(data=df, x='TEST_LOSS', y = 'MIN_GRAD_SV3')
    ax.axhline(-0.262, c='r', linestyle='--', label = 'True System')
    ax.set_xscale('log')
    ax.set_xlabel('Log(MSE)')
    ax.set_ylabel('Minimum Gradient of $\hat\sigma_3$')
    ax.set_title('Minimum Gradient of $\hat\sigma_3$ Against Test MSE')
    ticks = ax.get_yticks()
    ax.set_yticks(sorted(list(ticks) + [-0.262]))
    ax.legend()

    plt.show()
    plt.close('all')

if __name__ == "__main__":
    plot_mse_min_grad(df=pd.read_csv(rf'.\results\lorenz63.csv'), activation='tanh', width=16)