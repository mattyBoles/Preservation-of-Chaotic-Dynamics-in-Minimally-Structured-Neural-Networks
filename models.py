import torch
from torch import nn



class tanh_model(torch.nn.Module):
    def __init__(self, hidden_units, activation, input_size: int = 3, RANDOM_SEED=None):
        super().__init__()
        if RANDOM_SEED is not None:
            torch.manual_seed(RANDOM_SEED)

        if isinstance(activation, nn.Module):
            self.activation = activation
        elif isinstance(activation, type) and issubclass(activation, nn.Module):
            self.activation = activation()
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError(f"Unknown activation: {activation}")
                    
        self.linear1 = torch.nn.Linear(in_features=input_size, out_features=hidden_units)
        self.linear2 = torch.nn.Linear(in_features=hidden_units, out_features=input_size)


    def forward(self, x):

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x
    


class avg_euclidean_error(torch.nn.Module):
    def __init__(self,
                 mean,
                 std):
        super().__init__()
        self.mean = mean
        self.std = std
    def forward(self, pred, target):
      pred = (pred * self.std.to(pred.device)) + self.mean.to(pred.device)
      target = (target * self.std.to(pred.device)) + self.mean.to(pred.device)
      error = torch.linalg.norm(pred - target, axis = 1)
      return torch.mean(error)


class parameterised_beta_model(torch.nn.Module):
    def __init__(self, hidden_units, RANDOM_SEED = None):
        super().__init__()
        if RANDOM_SEED is not None:
            torch.manual_seed(RANDOM_SEED)

        self.linear1 = torch.nn.Linear(in_features = 3, out_features = hidden_units)
        self.linear2 = torch.nn.Linear(in_features = hidden_units, out_features = 3)

        betas = torch.ones(hidden_units)
        alphas = torch.ones(hidden_units)

        self.softplus = parameterised_softplus(hidden_units=hidden_units, alphas = alphas, betas = betas)

    def forward(self, x):
        x = self.linear1(x)
        x = self.softplus(x)
        x = self.linear2(x)
        return x


class parameterised_softplus(torch.nn.Module):
    def __init__(self, hidden_units, alphas,  betas, threshold=20):
        super().__init__()
        self.betas = torch.nn.Parameter(betas)
        self.alphas = torch.nn.Parameter(alphas)
        self.hidden_units = hidden_units
        self.threshold = threshold

    def forward(self, x):
        betas = torch.where(
            torch.abs(self.betas) < 1e-4,
            torch.where(self.betas >= 0, 1e-4, -1e-4),
            self.betas
        )

        bx = betas * x


        return torch.where(
            bx > self.threshold,
            x,
            torch.log1p(torch.exp(bx)) / self.alphas
        )


class WeightedMSELoss(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.register_buffer("std", std)

    def forward(self, preds, targets):
        err = preds - targets
        mse = torch.mean(err**2, dim=0) * self.std.squeeze()**2
        return torch.mean(mse)