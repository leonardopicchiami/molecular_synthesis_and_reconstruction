############################################################################################################
#                                                                                                          #
# FileName    [convolutional_encoder.py]                                                                   #
#                                                                                                          #
# PackageName [models.encoder]                                                                             #
#                                                                                                          #
# Synopsis    [This file contains the class that models the convolutional neural network that plays        #
#              the role of encoder. It inherits the Pytorch nn.Module class by defining the various        #
#              layers in the constructor and overriding the forward method to define the actual behavior   #
#              of the network during training.]                                                            #
#                                                                                                          #
# Author      [Leonardo Picchiami]                                                                         #
#                                                                                                          #
############################################################################################################



import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    '''
    Class that models the convolutional neural network that plays the role of encoder. 
    It inherits the Pytorch nn.Module class by defining the various layers in the constructor and overriding 
    the forward method to define the actual behavior of the network during training.

    Attributes
    ----------
    conv1 : nn.Conv2d
        First layer that performs the 2D convolution.

    conv2 : nn.Conv2d
        Second layer that performs the 2D convolution.

    conv3 : nn.Conv2d
        Third layer that performs the 2D convolution.

    fc_mu : nn.Linear
        Linear layer that calculates the mean of the probability distribution of the latent space of the autoncoder.

    fc_logvar : nn.Linear
        Linear layer that calculates the log-variance of the probability distribution of the latent space of the autoncoder.

    activation : nn.ReLU or nn.SeLU
        Neural network activation function which can be ReLU or SeLU depending on what is chosen when the object is initialized.

    dropout : nn.Dropout
        Dropout layer to be applied to each output of the convolutional linear map (except the last layer).
    '''

    
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


        '''
        Parameters
        ----------
        in_channels : int
            Input channel size.

        one_hot_columns : int
            Column size of each one hot encoding matrix.
        
        one_hot_rows : int
            Row size of each one hot encoding matrix.

        hidden_channels : int
            Reference hidden channel size of the convolutional layers.

        latent_dim : int
            Latent space dimension size of autoencoder.
        
        kernel_size : int
            Square kernel size.

        stride : int
            Size of the stride in the convolution.

        padding : int
            Size of the padding around the matrix.

        activation_function : str
            String aimed at choosing whether to use the ReLU or SeLU activation function.

        dropout : nn.Dropout
            Dropout layer to be applied to each output of the convolutional linear map (except the last layer).
        '''
        
        
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
            self.__activation = nn.SeLU()


        self.__dropout = dropout


        
    def forward(self, x : torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        Overriding the forward method which does:
            1.   The application of the 2d convolution as a linear map followed by the non-linear 
                 application of the ReLU (or SeLU) function.
            2.   The application of the 2d convolution as a linear map followed by the non-linear 
                 application of the ReLU (or SeLU) function.
            3.   The application of the 2d convolution as a linear map followed by the non-linear
                 application of the ReLU (or SeLU) function.
            4.1. The application of the linear layer as linear map for the latent mean calculation.
            4.2. The application of the linear layer as linear map for the log-variance calculation.


        Parameters
        ----------
        x : torch.Tensor
            Input data sample from the dataset.

        Returns
        -------
        (x_mu, x_logvar) : (torch.Tensor, torch.Tensor)
            Tuple of two tensors:
                - mean of the probability distribution of the latent space of the input sample x.
                - log-variance of the probability distribution of the latent space of the input sample x.
        '''
        
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


    


    
        
        

        
