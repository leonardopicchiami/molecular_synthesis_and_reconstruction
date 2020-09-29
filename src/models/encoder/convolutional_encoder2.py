########################################################################################
#                                                                                      #
# FileName    [convolutional_encoder2.py]                                              #
#                                                                                      #
# PackageName [models.encoder]                                                         #
#                                                                                      #
# Synopsis [This file is a mirror of convolutional_encoder.py used in the notebook.    #
#           For documentation read that file.]                                         #
#                                                                                      #
# Author   [Leonardo Picchiami]                                                        #
#                                                                                      #
########################################################################################


import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels : int, one_hot_columns : int,
                                          one_hot_rows : int,
                                          hidden_channels : int,
                                          latent_dim : int,
                                          kernel_size : int,
                                          stride : int,
                                          padding : int,
                                          activation_function : str,
                                          dropout : nn.Dropout2d = None) -> None:

        super().__init__()


        
        #Two convolutional layer
        self.__conv1 = nn.Conv2d(in_channels = in_channels, 
                               out_channels = int(hidden_channels/2), 
                               kernel_size = kernel_size, 
                               stride = stride, 
                               padding = padding)

        self.__conv2 = nn.Conv2d(in_channels = int(hidden_channels/2), 
                               out_channels = hidden_channels, 
                               kernel_size = kernel_size, 
                               stride = stride, 
                               padding = padding)

        self.__conv3 = nn.Conv2d(in_channels = hidden_channels, 
                               out_channels = hidden_channels*2, 
                               kernel_size = kernel_size, 
                               stride = stride, 
                               padding = padding)


        self.__fc_mu = nn.Linear(in_features = hidden_channels*2*one_hot_rows*one_hot_columns,
                                 out_features = latent_dim)
        
        self.__fc_logvar = nn.Linear(in_features = hidden_channels*2*one_hot_rows*one_hot_columns,
                                 out_features = latent_dim)

        #Activation function
        if activation_function == "relu":
            self.__activation = nn.ReLU()
        elif activation_function == "selu":
            self.__activation == nn.SeLU()


        self.__dropout = dropout


    def forward(self, x : torch.Tensor) -> (torch.Tensor, torch.Tensor):

        x_out = self.__activation(self.__conv1(x))
        if self.__dropout:
            x_out = self.__dropout(x_out)
            
        x_out = self.__activation(self.__conv2(x_out))
        if self.__dropout:
            x_out = self.__dropout(x_out)
            
        x_out = self.__activation(self.__conv3(x_out))
        if self.__dropout:
            x_out = self.__dropout(x_out)

        x_out = x_out.contiguous().view(x_out.shape[0], -1)
        x_mu = self.__fc_mu(x_out)
        x_logvar = self.__fc_logvar(x_out)
        return (x_mu, x_logvar)


    


    
        
        

        
