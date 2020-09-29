################################################################################
#                                                                              #
# FileName     [utils.py]                                                      #
#                                                                              #
# PackageName  [utils]                                                         #
#                                                                              #
# Synopsis     [File that contains a set of utility functions.]                #
#                                                                              #
# Author       [Leonardo Picchiami]                                            #
#                                                                              #
################################################################################



from typing import Callable, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import pandas as pd

import one_hot_decoding as hot_dec



def vae_loss_function(recon_x : torch.Tensor, x : torch.Tensor, mu : torch.Tensor, logvar : torch.Tensor) -> float:
    '''
    Loss function for the variational autoencoder.

    The loss function used is the binary cross entropy applied to the reconstructed sample x and the original one, 
    the result of which is added to the KL divergence, which makes it a varational autoencoder.

    Returns
    -------
    recon_loss + kldivergence : float
        Loss value calculated for the variational autoencoder.
    '''
    
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kldivergence



def refresh_bar(bar : tqdm, desc : str) -> None:
    '''
    Function that updates the tqdm bar during the model training phase.

    Parameters
    ----------
    bar : tqdm 
        The training bar to be updated.

    desc : str
        The description that must appear in the update of the tqdm bar.
    '''
    
    bar.set_description(desc)
    bar.refresh()


    
def make_averager() -> Callable[[Optional[float]], float]:
    ''' 
    Returns a function that maintains a running average.

    Returns
    -------
        The running average function.
    '''

    count = 0
    total = 0

    def averager(new_value: Optional[float]) -> float:
        '''
        Running averager
        
        Params
        ------
        new_value: float or None 
            Number to add to the running average; if None returns the current average.

        Returns
        -------
        average : float
            The current average.
        '''
        
        nonlocal count, total
        if new_value is None:
            return total / count if count else float("nan")
        
        count += 1
        total += new_value
        return total / count

    return averager



def model_train(model : nn.Module, train_loader : DataLoader,
                                   optimizer : torch.optim.Adam,
                                   num_epochs : int,
                                   loss_function : Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float],
                                   device : str) -> List[float]:

    '''
    Function for training a given input model.

    Parameters
    ----------
    model : nn.Model
        Model, i.e. varational autoencoder, which needs to be trained.

    train_loader : DataLoader
        DataLoder of the custom training set used training utilities such as mini-batches and shuffling.

    optimizer : torch.optim.Adam
        Adam optimizer for the recalculation of the neural network weights by minimizing the calculated loss.

    num_epochs : int
        Number of training epochs of the model.

    loss_function : Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float]
        Custom loss input function for error calculation.

    device : str
        Device on which the computation is performed. Typically it is either cpu or cuda (gpu).


    Returns
    -------
    running_rec_loss : list
        Returns the list of values relating to the average error at the end of each epoch.
    
    '''
    
    running_rec_loss = []
    loss = 0
    model.train()
    
    tqdm_bar = tqdm(range(1, num_epochs + 1), desc="epoch [loss: ...]")
    for epoch in tqdm_bar:
        train_loss_averager = make_averager()
        batch_bar =  tqdm(train_loader, leave = False, desc = 'batch', total = len(train_loader))
        
        for batch in batch_bar:
            batch = batch.float()
            batch = batch.to(device)
            batch_reconstructed, latent_mu, latent_logvar = model(batch)
            loss = loss_function(batch_reconstructed, batch, latent_mu, latent_logvar)

            #Backpropagation
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            refresh_bar(batch_bar, f"train batch [loss: {train_loss_averager(loss.item()):.3f}]")

    
        refresh_bar(tqdm_bar, f"epoch [loss: {train_loss_averager(None):.3f}]")
        running_rec_loss.append(train_loss_averager(None))

    return running_rec_loss
