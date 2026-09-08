import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import Callable, Tuple


def train_epoch(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: Callable,
                optimiser: torch.optim.Optimizer,
                err_fn: Callable,
                device: torch.device)-> Tuple[float, float]:
    
    model = model.to(device)

    model.train()

    epoch_err = 0.0
    epoch_loss = 0.0
    epoch_preds = []

    n_batches = len(dataloader)

    for idx, (inputs, targets) in enumerate(dataloader):
        
        targets = targets.float().to(device)
        inputs = inputs.float().to(device)

        def closure():
            optimiser.zero_grad()

            preds = model(inputs)
    
            loss = loss_fn(preds, targets)
            loss.backward()
            return loss


        loss = optimiser.step(closure)

        epoch_loss += loss.item() 
        epoch_preds.append(model(inputs))

    epoch_preds = torch.cat(epoch_preds, dim=0)
    err = err_fn(epoch_preds, targets)
    epoch_err += err.item()

    epoch_loss /= n_batches
    epoch_err /= n_batches

    return epoch_loss, epoch_err

def val_epoch(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: Callable,
              err_fn: Callable,
              device: torch.device)-> Tuple[float, float]:
    
    model = model.to(device)

    model.eval()

    epoch_err = 0.0
    epoch_loss = 0.0
    epoch_preds = []

    n_batches = len(dataloader)
    with torch.inference_mode():
        for idx, (inputs, targets) in enumerate(dataloader):
            
            targets = targets.float().to(device)
            inputs = inputs.float().to(device)

            def closure():

                preds = model(inputs)
        
                loss = loss_fn(preds, targets)
                return loss
           
            loss = closure()
           
               
            epoch_loss += loss.item() 
            epoch_preds.append(model(inputs))
           
        epoch_preds = torch.cat(epoch_preds, dim=0)
        err = err_fn(epoch_preds, targets)
        epoch_err += err.item()
           
        epoch_loss /= n_batches
        epoch_err /= n_batches
           
        return epoch_loss, epoch_err


def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          loss_fn: Callable,
          optimiser: torch.optim.Optimizer,
          err_fn: Callable,
          NUM_EPOCHS: int,
          device: torch.device)->dict[str: list]:
    
    results = {
        'train_loss': [],
        'train_err': [],
        'val_loss': [],
        'val_err': [],
        'model_statedict': [],
    }
    pbar = tqdm(range(1, NUM_EPOCHS+1), desc="Training", colour="green")
    for epoch in pbar:
        pbar.set_description(f"Training Epoch: {epoch}/{NUM_EPOCHS}")
        train_loss, train_err = train_epoch(model = model,
                                            dataloader = train_loader,
                                            loss_fn = loss_fn,
                                            optimiser = optimiser,
                                            err_fn = err_fn,
                                            device = device)
        pbar.set_description(f"Validating Epoch: {epoch}/{NUM_EPOCHS}")
        val_loss, val_err = val_epoch(model = model,
                                      dataloader = val_loader,
                                      loss_fn = loss_fn,
                                      err_fn = err_fn,
                                      device = device)
        
        results['train_loss'].append(train_loss)
        results['train_err'].append(train_err)
        results['val_loss'].append(val_loss)
        results['val_err'].append(val_err)
        results['model_statedict'].append(deepcopy(model.state_dict()))

        if epoch % 10 == 0:
            print(f'| Epoch {epoch} |\n| Train Loss : {train_loss} | Train Average Euclidean Distance: {train_err} |\n| Val Loss : {val_loss} | Val Average Euclidean Distance: {val_err} |')

    return results    
    

def evaluate(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         loss_fn: Callable,
         err_fn: Callable,
         device: torch.device) -> Tuple[float, float]:
    
    
    model = model.to(device)

    model.eval()
    epoch_preds = []

    epoch_err = 0.0
    epoch_loss = 0.0
    n_batches = len(dataloader)
    with torch.inference_mode():
        for idx, (inputs, targets) in enumerate(dataloader):
            
            targets = targets.float().to(device)
            inputs = inputs.float().to(device)

            def closure():
            
                preds = model(inputs)
                    
                loss = loss_fn(preds, targets)
                return loss
                       
            loss = closure()
                

            epoch_loss += loss.item() 
            epoch_preds.append(model(inputs))
           
        epoch_preds = torch.cat(epoch_preds, dim=0)
        err = err_fn(epoch_preds, targets)
        epoch_err += err.item()
           
        epoch_loss /= n_batches
        epoch_err /= n_batches
           
        return epoch_loss, epoch_err



