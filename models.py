import torch
from torch import nn

class Lorenz_model(torch.nn.Module):
    def __init__(self, hidden_units:int, activation, input_size: int = 3, RANDOM_SEED=None):
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
                 mean: torch.Tensor,
                 std: torch.Tensor):
        '''
        Calculates the euclidean distance between pred and targets, better for visualisation when we normalise.

        Inputs:
            mean (torch.Tensor): The mean of the training set as a vector, as retruened by the Datasets init.
            std (torch.Tensor): The std of the training set as a vector, as retruened by the Datasets init.

        '''
        super().__init__()
        self.mean = mean
        self.std = std
    def forward(self, preds: torch.Tensor, targets:torch.Tensor)-> torch.Tensor:
        '''
        Inputs:
            preds (torch.Tensor): Model predictions in shape (batch, input_size).
            targets (torch.Tensor): true values in shape (batch, input_size).
        
        Returns:
            torch.Tensor: The avergae euclidean distance over inputs from true values, 
        '''
        preds = (preds * self.std.to(preds.device)) + self.mean.to(preds.device)
        targets = (targets * self.std.to(preds.device)) + self.mean.to(preds.device)
        error = torch.linalg.norm(preds - targets, axis = 1)
        return torch.mean(error)


class WeightedMSELoss(nn.Module):
    def __init__(self, std:torch.Tensor):
        '''
        MSE split over dimensions.
        '''
        super().__init__()
        self.register_buffer("std", std)

    def forward(self,
                preds: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
            preds (torch.Tensor): Model predictions in shape (batch, input_size).
            targets (torch.Tensor): true values in shape (batch, input_size).
        
        Returns:
            torch.Tensor: The MSE in each dimension.
        '''
        err = preds - targets
        mse = torch.mean(err**2, dim=0) * self.std.squeeze()**2
        return torch.mean(mse)