############################################################################################################
#                                                                                                          #
# FileName    [gru_decoder.py]                                                                             #
#                                                                                                          #
# PackageName [models.decoder]                                                                             #
#                                                                                                          #
# Synopsis    [This file contains the class that models the GRU + Linear neural network that plays         #
#              the role of decoder. It inherits the Pytorch nn.Module class by defining the various        #
#              layers in the constructor and overriding the forward method to define the actual behavior   #
#              of the network during training.]                                                            #
#                                                                                                          #
# Author      [Leonardo Picchiami]                                                                         #
#                                                                                                          #
############################################################################################################


import torch
import torch.nn as nn


class GRULinearDecoder(nn.Module):
    '''
    Class that models the GRU + Linear neural network that plays the role of decoder. 
    It inherits the Pytorch nn.Module class by defining the various layers in the constructor 
    and overriding the forward method to define the actual behavior of the network during training.

    Attributes
    ----------
    one_hot_columns : int
        Column size of each one hot encoding matrix. 

    one_hot_rows : int
        Row size of each one hot encoding matrix.
    
    hidden_dim : int
        The number of features in the hidden state h.

    fc : nn.Linear
        Linear layer that performs linear application from the latent space necessary for the autoencoder.

    gru : nn.GRU
        GRU multi-layer that performs the intermediate decoding task on the output of the linear map of the linear layer 
        on the representation of the latent space.

    fc1 : nn.Layer
        Linear layer that makes a linear map without increasing or decreasing the number of features.
    '''
    
    
    def __init__(self, one_hot_columns : int, one_hot_rows : int,
                                              hidden_dim : int,
                                              linear_dim : int,
                                              latent_dim : int,
                                              num_layers : int = 1,
                                              dropout : float = 0,
                                              isBidirectional : bool = False) -> None:


        '''
        Parameters
        ----------
        one_hot_columns : int
            Column size of each one hot encoding matrix. 

        one_hot_rows : int
            Row size of each one hot encoding matrix.
    
        hidden_dim : int
            The number of features in the hidden state h.

        linear_dim : int
            Size of the input of the second linear layer.

        latent_dim : int
            Latent space dimension size of autoencoder.
        
        num_layers : int
            Number of layers of the GRU multi-layer.

        dropout : float
            If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer,
            with dropout probability equal to dropout input.
        
        isBidirectional : bool
            if True, the GRU multi-layer becomes a bidirectional GRU multi-layer
        
        '''

        
        super().__init__()

        self.__one_hot_columns = one_hot_columns
        self.__one_hot_rows = one_hot_rows
        self.__hidden_dim = hidden_dim
        

        #Linear operation from latent space
        self.__fc = nn.Linear(in_features = latent_dim, 
                                 out_features = linear_dim)

        #GRU layer
        self.__gru = nn.GRU(hidden_dim, one_hot_rows, num_layers, dropout = dropout, bidirectional = isBidirectional)


        #Linear Layer to one-hot-encoding output
        self.__fc1 = nn.Linear(in_features = one_hot_columns*one_hot_rows, 
                                 out_features = one_hot_columns*one_hot_rows) 


        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Overriding the forward method which does:
            1. Application of the linear map of the linear layer.
            2. The application of the GRU on the output of the linear layer applied on the representation of the latent space.
            3. The application of the second linear layer application followed by the sigmoid fuction applicaton returning the decoding as output.


        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing the latent space to be reconstructed.

        Returns
        -------
        x_out : torch.Tensor
            Output tensor containing the decoder otuput.
        
        '''
        
        x_rec = self.__fc(x)
        x_rec = x_rec.contiguous().view(x_rec.shape[0], self.__one_hot_columns, self.__hidden_dim)

        x_rec, h_c = self.__gru(x_rec)
        x_rec = x_rec.view(x_rec.size(0), -1)

        x_rec = torch.sigmoid(self.__fc1(x_rec))
        x_rec = x_rec.view(x_rec.size(0), -1, self.__one_hot_columns, self.__one_hot_rows)
        return x_rec
